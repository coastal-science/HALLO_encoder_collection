import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from lightly.models.modules import MoCoProjectionHead, SimCLRProjectionHead
from lightly.models.utils import deactivate_requires_grad

from encoder_pipeline.model_trainer.config import (
    ClassifierConfig, EfficientNetVariant, MoCoConfig, MoCoV3Config, ResNetVariant, SimCLRConfig, backbone,
)
from encoder_pipeline.model_trainer.perch_v2 import PerchV2Backbone, load_perch_v2_backbone
from encoder_pipeline.model_trainer.ssamba import SSAMBABackbone


class ResNetBackbone(nn.Module):
    """torchvision ResNet family"""

    def __init__(self, variant: ResNetVariant) -> None:
        super().__init__()
        resnet = getattr(tv_models, variant)()
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.out_features = resnet.fc.in_features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(start_dim=1)


class EfficientNetBackbone(nn.Module):
    """torchvision EfficientNet family"""

    def __init__(self, variant: EfficientNetVariant) -> None:
        super().__init__()
        effnet = getattr(tv_models, variant)()
        first_conv = effnet.features[0][0]
        effnet.features[0][0] = nn.Conv2d(
            1, first_conv.out_channels, kernel_size=first_conv.kernel_size,
            stride=first_conv.stride, padding=first_conv.padding, bias=False,
        )
        self.features = effnet.features
        self.out_features = effnet.classifier[1].in_features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(start_dim=1)


def build_backbone(name: backbone) -> nn.Module:
    if name.startswith("efficientnet"):
        return EfficientNetBackbone(name)
    return ResNetBackbone(name)


class SimCLRModel(nn.Module):
    def __init__(self, config: SimCLRConfig) -> None:
        super().__init__()
        self.backbone = build_backbone(config.backbone_name)
        self.projection_head = SimCLRProjectionHead(
            self.backbone.out_features, config.projection_hidden_dim, config.projection_out_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_head(self.backbone(x))


class MoCoModel(nn.Module):
    """Query encoder (backbone + projection head) trained by gradient, plus a
    momentum-updated copy that produces the keys."""

    def __init__(self, config: MoCoConfig) -> None:
        super().__init__()
        self.backbone = build_backbone(config.backbone_name)
        self.projection_head = MoCoProjectionHead(
            self.backbone.out_features, config.projection_hidden_dim, config.projection_out_dim,
        )
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_head(self.backbone(x))

    @torch.no_grad()
    def forward_momentum(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_head_momentum(self.backbone_momentum(x))


class MoCoV3Model(nn.Module):
    """MoCo v3 query encoder: backbone -> 3-layer projection head -> 2-layer
    prediction head, all trained by gradient. The momentum-updated backbone +
    projection head (no prediction head) produce the keys."""

    def __init__(self, config: MoCoV3Config) -> None:
        super().__init__()
        self.backbone = build_backbone(config.backbone_name)
        self.projection_head = MoCoProjectionHead(
            self.backbone.out_features, config.projection_hidden_dim, config.projection_out_dim,
            num_layers=3, batch_norm=True,
        )
        self.prediction_head = MoCoProjectionHead(
            config.projection_out_dim, config.prediction_hidden_dim, config.projection_out_dim,
            num_layers=2, batch_norm=True,
        )
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prediction_head(self.projection_head(self.backbone(x)))

    @torch.no_grad()
    def forward_momentum(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_head_momentum(self.backbone_momentum(x))


def load_backbone(backbone: nn.Module, checkpoint_path: str) -> None:
    """Loads the backbone of a prior model_trainer checkpoint (any paradigm's model.backbone) into backbone, leaving
    out every head, so it can start a model with another class list. An SSAMBA checkpoint pretrained on another
    patch grid (spectrogram size) has its position embeddings resized bilinearly to backbone's grid."""
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"].backbone
    state = source.state_dict()
    if isinstance(backbone, SSAMBABackbone) and state["v.pos_embed"].shape != backbone.v.pos_embed.shape:
        cls_pos, grid = state["v.pos_embed"][:, :1], state["v.pos_embed"][:, 1:]
        grid = grid.reshape(1, source.p_f_dim, source.p_t_dim, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(backbone.p_f_dim, backbone.p_t_dim), mode="bilinear", align_corners=False)
        state["v.pos_embed"] = torch.cat([cls_pos, grid.flatten(2).transpose(1, 2)], dim=1)
    backbone.load_state_dict(state)


def build_ssamba_backbone(config: ClassifierConfig) -> SSAMBABackbone:
    """SSAMBA encoder for a classifier, initialised from ssamba.pretrained_path if set."""
    assert config.ssamba is not None, "model_trainer.classifier.ssamba is required for an ssamba_* backbone_name"
    ssamba_backbone = SSAMBABackbone(config.backbone_name, config.ssamba)
    if config.ssamba.pretrained_path is not None:
        load_backbone(ssamba_backbone, config.ssamba.pretrained_path)
    return ssamba_backbone


def build_perch_v2_backbone(config: ClassifierConfig) -> PerchV2Backbone:
    """Perch v2's EfficientNet-B3 with Google's weights (converted by scripts/perch/convert_perch_v2_to_torch.py)."""
    assert config.perch_v2 is not None, "model_trainer.classifier.perch_v2 is required for backbone_name perch_v2"
    perch = config.perch_v2
    return load_perch_v2_backbone(perch.pretrained_path, input_layout=perch.input_layout,
                                  drop_path_rate=perch.drop_path_rate, freeze_batchnorm=perch.freeze_batchnorm)


class ClassifierModel(nn.Module):
    def __init__(self, config: ClassifierConfig, num_classes: int) -> None:
        super().__init__()
        if config.backbone_name.startswith("ssamba"):
            self.backbone = build_ssamba_backbone(config)
        elif config.backbone_name == "perch_v2":
            self.backbone = build_perch_v2_backbone(config)
        else:
            self.backbone = build_backbone(config.backbone_name)
        if config.init_backbone_checkpoint is not None:
            load_backbone(self.backbone, config.init_backbone_checkpoint)
        self.classifier = nn.Linear(self.backbone.out_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))
