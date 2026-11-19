"""Audio-JEPA (model_trainer/audio_jepa, vendored from LudovicTuncay/Audio-JEPA) on encoder_pipeline's spectrograms.

AudioJEPABackbone wraps upstream's VisionTransformer as a classifier / embedding backbone; AudioJEPAModel is upstream's
JEPAModule without Lightning: context encoder, EMA target encoder (deepcopy of the context encoder), ViT predictor and
loss, with JEPAModule.model_step as forward and MAWeightUpdate.update_weights as update_target.
"""
import copy
from math import cos, pi
from typing import Optional, Union

import torch
import torch.nn as nn

from encoder_pipeline.model_trainer.audio_jepa.loss import Loss
from encoder_pipeline.model_trainer.audio_jepa.mask_utils import apply_masks
from encoder_pipeline.model_trainer.audio_jepa.vision_transformer import VisionTransformer, VisionTransformerPredictor
from encoder_pipeline.model_trainer.config import (
    AudioJEPAClassifierConfig, AudioJEPAConfig, AudioJEPAEncoderConfig, AudioJEPAVariant,
)

# (embed_dim, depth, num_heads), as upstream's vit_tiny / vit_small / vit_base. The encoder itself is built like
# upstream's configs/model/jepa.yaml builds it (VisionTransformer directly, so its default LayerNorm and qkv_bias).
AUDIO_JEPA_VARIANTS: dict[str, tuple[int, int, int]] = {
    "audio_jepa_vit_tiny": (192, 12, 3),
    "audio_jepa_vit_small": (384, 12, 6),
    "audio_jepa_vit_base": (768, 12, 12),
}


class AudioJEPABackbone(nn.Module):
    """Upstream's VisionTransformer over (time, freq) patches of a (B, 1, n_mels, frames) spectrogram batch;
    forward returns the mean patch token."""

    def __init__(self, variant: AudioJEPAVariant, config: AudioJEPAEncoderConfig) -> None:
        super().__init__()
        embed_dim, depth, num_heads = AUDIO_JEPA_VARIANTS[variant]
        self.input_fdim, self.input_tdim = config.input_fdim, config.input_tdim
        self.patch_t, self.patch_f = config.patch_size
        self.grid_t, self.grid_f = config.input_tdim // self.patch_t, config.input_fdim // self.patch_f
        self.input_norm = config.input_norm
        self.out_features = embed_dim
        self.encoder = VisionTransformer(
            input_size=(self.grid_t * self.patch_t, self.grid_f * self.patch_f), patch_size=tuple(config.patch_size),
            in_chans=1, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=4.0,
            drop_path_rate=config.drop_path_rate,
        )

    @property
    def num_patches(self) -> int:
        return self.grid_t * self.grid_f

    def prepare(self, specs: torch.Tensor) -> torch.Tensor:
        """(B, 1, F, T) preprocessor spectrograms -> (B, 1, T', F') upstream layout, normalised and cropped to whole
        patches."""
        assert specs.shape[-2:] == (self.input_fdim, self.input_tdim), (
            f"audio_jepa expects ({self.input_fdim}, {self.input_tdim}) spectrograms, got {tuple(specs.shape[-2:])}"
        )
        if self.input_norm == "per_clip":
            mean = specs.mean(dim=(-2, -1), keepdim=True)
            std = specs.std(dim=(-2, -1), keepdim=True)
            specs = (specs - mean) / (std + 1e-5)
        specs = specs.transpose(-2, -1)
        return specs[..., : self.grid_t * self.patch_t, : self.grid_f * self.patch_f]

    def tokens(self, specs: torch.Tensor, context_masks: Optional[list[torch.Tensor]] = None) -> torch.Tensor:
        """Patch tokens (len(context_masks) * B, kept patches, D), or (B, num_patches, D) without masks."""
        return self.encoder(self.prepare(specs), context_masks)

    def forward(self, specs: torch.Tensor) -> torch.Tensor:
        return self.tokens(specs).mean(dim=1)


class AudioJEPAModel(nn.Module):
    """Upstream JEPAModule's networks. `backbone` (what load_backbone exports to fine-tunes and embeddings) is the
    target or the context encoder, per config.downstream_encoder."""

    def __init__(self, config: AudioJEPAConfig) -> None:
        super().__init__()
        self.downstream_encoder = config.downstream_encoder
        self.context_encoder = AudioJEPABackbone(config.backbone_name, config)
        # jepa_module.py: target encoder as a moving average of the context encoder, no gradients
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.predictor = VisionTransformerPredictor(
            num_patches_h=self.context_encoder.grid_t, num_patches_w=self.context_encoder.grid_f,
            embed_dim=self.context_encoder.out_features, predictor_embed_dim=config.predictor_embed_dim,
            depth=config.predictor_depth, num_heads=config.predictor_num_heads, mlp_ratio=4.0,
        )
        self.criterion = Loss(loss_type=config.loss_type, norm_pix_loss=config.norm_pix_loss)

    @property
    def backbone(self) -> AudioJEPABackbone:
        return self.target_encoder if self.downstream_encoder == "target" else self.context_encoder

    def forward(
        self, specs: torch.Tensor, context_masks: list[torch.Tensor], prediction_masks: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """JEPAModule.model_step: (loss, predicted target tokens)."""
        h = self.context_encoder.tokens(specs, context_masks)
        z_pred = self.predictor(h, context_masks, prediction_masks)
        with torch.no_grad():
            h_target = self.target_encoder.tokens(specs)
            h_target = apply_masks(h_target, prediction_masks)
        return self.criterion(z_pred, h_target), z_pred

    @torch.no_grad()
    def update_target(self, tau: float) -> None:
        """MAWeightUpdate.update_weights: target <- tau * target + (1 - tau) * context, parameter by parameter."""
        for (_, student_p), (_, teacher_p) in zip(
            self.context_encoder.named_parameters(), self.target_encoder.named_parameters(),
        ):
            teacher_p.data = tau * teacher_p.data + (1 - tau) * student_p.data


def ema_tau(step: int, total_steps: int, initial_tau: float, final_tau: float) -> float:
    """MAWeightUpdate.update_tau_cos: initial_tau at step 0, cosine-ramped to final_tau at total_steps."""
    return final_tau - (final_tau - initial_tau) * (cos(pi * step / total_steps) + 1) / 2


def _without_pos_embed(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drops the fixed 2-D sin-cos position embeddings (pos_embed / predictor_pos_embed: requires_grad=False buffers
    rebuilt from the patch grid), so weights load across spectrogram sizes."""
    return {k: v for k, v in state.items() if not k.endswith("pos_embed")}


def load_audio_jepa_state(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Strict load of every weight except the position embeddings, which module keeps for its own grid."""
    missing, unexpected = module.load_state_dict(_without_pos_embed(state), strict=False)
    missing = [k for k in missing if not k.endswith("pos_embed")]
    assert not missing and not unexpected, f"audio_jepa weights don't match: missing {missing}, unexpected {unexpected}"


def audioset_state(checkpoint_path: str, module: str) -> dict[str, torch.Tensor]:
    """One module's weights ('encoder', 'target_encoder' or 'predictor') from upstream's Lightning JEPA.ckpt."""
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    prefix = f"{module}."
    state = {k[len(prefix):]: v for k, v in bundle.get("state_dict", bundle).items() if k.startswith(prefix)}
    assert state, f"no {module}.* weights in {checkpoint_path}"
    return state


def init_from_audioset(model: Union[AudioJEPAModel, AudioJEPABackbone], checkpoint_path: str,
                       backbone_module: str = "encoder") -> None:
    """Loads upstream's AudioSet JEPA.ckpt: into a whole AudioJEPAModel (encoder -> context, target_encoder ->
    target, predictor -> predictor), or into one AudioJEPABackbone (backbone_module)."""
    if isinstance(model, AudioJEPABackbone):
        load_audio_jepa_state(model.encoder, audioset_state(checkpoint_path, backbone_module))
        return
    load_audio_jepa_state(model.context_encoder.encoder, audioset_state(checkpoint_path, "encoder"))
    load_audio_jepa_state(model.target_encoder.encoder, audioset_state(checkpoint_path, "target_encoder"))
    load_audio_jepa_state(model.predictor, audioset_state(checkpoint_path, "predictor"))


def build_audio_jepa_backbone(variant: AudioJEPAVariant, config: AudioJEPAClassifierConfig) -> AudioJEPABackbone:
    """Classifier backbone, from audio_jepa.audioset_checkpoint if set (pretrained_path is loaded by the caller,
    models.load_backbone, like every other paradigm's checkpoint)."""
    assert not (config.audioset_checkpoint and config.pretrained_path), (
        "set at most one of classifier.audio_jepa.audioset_checkpoint / pretrained_path"
    )
    backbone = AudioJEPABackbone(variant, config)
    if config.audioset_checkpoint is not None:
        init_from_audioset(backbone, config.audioset_checkpoint, config.audioset_checkpoint_module)
    return backbone
