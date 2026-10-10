"""SSAMBA (Shams et al. 2024): a bidirectional Vision Mamba (Vim) encoder over
spectrogram patches, pretrained by masked spectrogram patch modelling. Ported
from https://github.com/SiavashShams/ssamba (src/models/both_models.py) and
Vim's bimamba_type="v2" mixer, keeping their parameter names.

setup: only Mamba's compiled selective-scan kernel is needed (no Vim / mamba
fork checkout), built against the installed torch:

    git clone https://github.com/state-spaces/mamba
    CUDA_HOME=/usr/local/cuda MAMBA_FORCE_BUILD=TRUE MAMBA_KEEP_CUDA_BUILD=TRUE \
        pip install --no-deps --no-build-isolation ./mamba
"""
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_

from encoder_pipeline.model_trainer.config import SSAMBAConfig, SSAMBAEncoderConfig, SSAMBAVariant

try:
    import selective_scan_cuda
except ImportError:
    selective_scan_cuda = None

SSAMBA_VARIANTS: dict[str, dict[str, int]] = {
    "ssamba_tiny": {"embed_dim": 192, "depth": 24},
    "ssamba_small": {"embed_dim": 384, "depth": 24},
    "ssamba_base": {"embed_dim": 768, "depth": 24},
}


class SelectiveScanFn(torch.autograd.Function):
    """mamba_ssm.ops.selective_scan_interface.SelectiveScanFn, with z always given."""

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, z, delta_bias):
        B, C = B.unsqueeze(1), C.unsqueeze(1)
        out, x, out_z = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, True)
        ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
        return out_z

    @staticmethod
    def backward(ctx, dout):
        u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        du, ddelta, dA, dB, dC, dD, ddelta_bias, dz = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout.contiguous(), x, out, None, True, False,
        )
        return du, ddelta, dA, dB.squeeze(1), dC.squeeze(1), dD, dz, ddelta_bias


class BiMamba(nn.Module):
    """Vim's bimamba_type="v2" Mamba mixer: a shared in / out projection around
    a forward and a backward selective scan, whose outputs are averaged."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        if selective_scan_cuda is None:
            raise ImportError("selective_scan_cuda is not installed -- see the setup note in model_trainer/ssamba.py")
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        for suffix in ("", "_b"):
            self.add_module(f"conv1d{suffix}", nn.Conv1d(
                self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1,
            ))
            self.add_module(f"x_proj{suffix}", nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False))
            dt_proj = nn.Linear(self.dt_rank, self.d_inner)
            nn.init.uniform_(dt_proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
            # bias is the inverse softplus of a dt drawn log-uniformly from [1e-3, 1e-1]
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
            with torch.no_grad():
                dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            self.add_module(f"dt_proj{suffix}", dt_proj)
            A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
            self.register_parameter(f"A{suffix}_log", nn.Parameter(torch.log(A)))
            self.register_parameter(f"D{suffix}", nn.Parameter(torch.ones(self.d_inner)))

    def _scan(self, x: torch.Tensor, z: torch.Tensor, suffix: str) -> torch.Tensor:
        """One direction's conv -> input-dependent (dt, B, C) -> selective scan, on (B, d_inner, L)."""
        conv, x_proj, dt_proj = (getattr(self, f"{name}{suffix}") for name in ("conv1d", "x_proj", "dt_proj"))
        x = F.silu(conv(x)[..., : x.shape[-1]])
        dt, B, C = x_proj(x.transpose(1, 2)).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = (dt @ dt_proj.weight.t()).transpose(1, 2)
        A = -torch.exp(getattr(self, f"A{suffix}_log"))
        return SelectiveScanFn.apply(
            x.contiguous(), dt.contiguous(), A, B.transpose(1, 2).contiguous(), C.transpose(1, 2).contiguous(),
            getattr(self, f"D{suffix}"), z.contiguous(), dt_proj.bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # the scan kernel is fp32-only, so the mixer is kept out of autocast
        with torch.autocast(hidden_states.device.type, enabled=False):
            x, z = self.in_proj(hidden_states.float()).transpose(1, 2).chunk(2, dim=1)
            out = self._scan(x, z, "")
            out_b = self._scan(x.flip(-1), z.flip(-1), "_b")
            return self.out_proj(((out + out_b.flip(-1)) / 2).transpose(1, 2))


class VimBlock(nn.Module):
    """Pre-norm residual block around a BiMamba mixer."""

    def __init__(self, dim: int, drop_path: float) -> None:
        super().__init__()
        self.mixer = BiMamba(dim)
        self.norm = nn.RMSNorm(dim, eps=1e-5)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.mixer(self.norm(x)))


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int, fshape: int, tshape: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(fshape, tshape), stride=(fshape, tshape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionMamba(nn.Module):
    """Vim encoder: patch embedding, cls token + learned positions, VimBlocks, final norm."""

    def __init__(self, embed_dim: int, depth: int, num_patches: int, fshape: int, tshape: int, drop_path_rate: float) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(embed_dim, fshape, tshape)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.layers = nn.ModuleList(
            VimBlock(embed_dim, rate) for rate in np.linspace(0, drop_path_rate, depth).tolist()
        )
        self.norm_f = nn.RMSNorm(embed_dim, eps=1e-5)
        for layer in self.layers:
            # Mamba's prenorm-residual rescale of each block's output projection
            with torch.no_grad():
                layer.mixer.out_proj.weight /= math.sqrt(depth)


class SSAMBABackbone(nn.Module):
    """Vim encoder over non-overlapping fshape x tshape spectrogram patches;
    forward returns the mean patch token (SSAMBA's ft_avgtok)."""

    def __init__(self, variant: SSAMBAVariant, config: SSAMBAEncoderConfig) -> None:
        super().__init__()
        self.input_fdim, self.input_tdim = config.input_fdim, config.input_tdim
        self.fshape, self.tshape = config.fshape, config.tshape
        self.p_f_dim, self.p_t_dim = config.input_fdim // config.fshape, config.input_tdim // config.tshape
        self.num_patches = self.p_f_dim * self.p_t_dim
        self.out_features = SSAMBA_VARIANTS[variant]["embed_dim"]
        self.v = VisionMamba(
            self.out_features, SSAMBA_VARIANTS[variant]["depth"], self.num_patches,
            config.fshape, config.tshape, config.drop_path_rate,
        )

    def normalize(self, specs: torch.Tensor) -> torch.Tensor:
        """Standardizes each (B, 1, F, T) spectrogram to zero mean / unit variance."""
        assert specs.shape[-2:] == (self.input_fdim, self.input_tdim), (
            f"ssamba expects ({self.input_fdim}, {self.input_tdim}) spectrograms, got {tuple(specs.shape[-2:])}"
        )
        mean = specs.mean(dim=(-2, -1), keepdim=True)
        std = specs.std(dim=(-2, -1), keepdim=True)
        return (specs - mean) / (std + 1e-5)

    def encode(
        self, specs: torch.Tensor, mask_index: Optional[torch.Tensor] = None,
        mask_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """(B, num_patches, embed_dim) patch tokens of normalized specs; the
        (B, M) mask_index patches are swapped for mask_embed before encoding."""
        x = self.v.patch_embed(specs)
        if mask_index is not None:
            masked = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device).scatter_(1, mask_index, True)
            x = torch.where(masked.unsqueeze(-1), mask_embed.to(x.dtype), x)
        x = torch.cat((self.v.cls_token.expand(x.shape[0], -1, -1), x), dim=1) + self.v.pos_embed
        for layer in self.v.layers:
            x = layer(x)
        return self.v.norm_f(x)[:, 1:]

    def forward(self, specs: torch.Tensor) -> torch.Tensor:
        return self.encode(self.normalize(specs)).mean(dim=1)


class SSAMBAModel(nn.Module):
    """SSAMBABackbone + SSAMBA's masked spectrogram patch modelling heads:
    cpredlayer picks each masked patch out of the clip's other masked patches
    (InfoNCE), gpredlayer reconstructs it (MSE). Both heads read one masked
    pass, where the reference repo runs a separately-masked pass per objective."""

    def __init__(self, config: SSAMBAConfig) -> None:
        super().__init__()
        self.backbone = SSAMBABackbone(config.backbone_name, config)
        assert config.mask_patch < self.backbone.num_patches, (
            f"mask_patch={config.mask_patch} must be below the {self.backbone.num_patches} patches per clip"
        )
        self.mask_patch = config.mask_patch
        self.cluster = config.cluster
        embed_dim, patch_dim = self.backbone.out_features, config.fshape * config.tshape
        self.cpredlayer = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, patch_dim))
        self.gpredlayer = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, patch_dim))
        self.unfold = nn.Unfold(kernel_size=(config.fshape, config.tshape), stride=(config.fshape, config.tshape))
        self.mask_embed = nn.Parameter(nn.init.xavier_normal_(torch.zeros(1, 1, embed_dim)))

    def gen_maskid_patch(self) -> np.ndarray:
        """mask_patch patch ids for one clip, drawn as square clusters of side 3-5
        when self.cluster is set, else uniformly."""
        num_patches = self.backbone.num_patches
        if not self.cluster:
            return np.random.choice(num_patches, self.mask_patch, replace=False)
        side = np.random.randint(3, 6)
        offsets = (np.arange(side)[:, None] * self.backbone.p_t_dim + np.arange(side)[None, :]).ravel()
        mask_id: set[int] = set()
        while len(mask_id) < self.mask_patch:
            candidates = np.random.randint(num_patches) + offsets
            mask_id.update(candidates[candidates < num_patches].tolist())
        return np.fromiter(mask_id, dtype=np.int64)[: self.mask_patch]

    def forward(self, specs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(nce, mse, acc) of the masked-patch objectives on a (B, 1, F, T) batch."""
        specs = self.backbone.normalize(specs)
        mask_index = torch.from_numpy(
            np.stack([self.gen_maskid_patch() for _ in range(specs.shape[0])])
        ).to(specs.device)
        tokens = self.backbone.encode(specs, mask_index, self.mask_embed)
        tokens = tokens.gather(1, mask_index.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])).float()
        patches = self.unfold(specs).transpose(1, 2)
        target = patches.gather(1, mask_index.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))
        # row = true patch, col = prediction; the matching prediction sits on the diagonal
        logits = target @ self.cpredlayer(tokens).transpose(1, 2)
        nce = -F.log_softmax(logits, dim=-1).diagonal(dim1=1, dim2=2).mean()
        acc = (logits.argmax(dim=1) == torch.arange(self.mask_patch, device=logits.device)).float().mean()
        mse = F.mse_loss(self.gpredlayer(tokens), target)
        return nce, mse, acc
