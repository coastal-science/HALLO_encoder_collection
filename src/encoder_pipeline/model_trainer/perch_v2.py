"""Perch v2 (Google's bird-vocalization-classifier, "Perch 2.0") as a trainable PyTorch model.

Google releases Perch v2 only as an inference-only TF SavedModel (a jax2tf export with gradients disabled). Its embedding
model is an EfficientNet-B3 on a log-mel spectrogram; this module re-implements both in PyTorch, mirroring Google's
Flax reference code (google-research/perch: chirp/models/efficientnet.py, chirp/models/layers.py and the
`SimpleMelspec` frontend), and `perch_v2_state_dict` maps the SavedModel's variables onto it. Weights come from
scripts/perch/convert_perch_v2_to_torch.py, which also checks this model against the SavedModel on real audio.

Details that have to be exactly right (each checked against the SavedModel, see the conversion script):
  * swish (SiLU) in every MBConv block, not only stem / head. chirp's "default" OpSet uses relu in the blocks, which
    loads the same weights without complaint but gives unrelated embeddings (cosine ~0.02).
  * BatchNorm eps 1e-5 (Flax's default; 1e-3 gives cosine ~0.77). Momentum 0.01 here = Flax's 0.99.
  * stride-2 depthwise convs pad (k//2 - 1 + n % 2, k//2) per spatial dim of size n, i.e. asymmetric on even sizes;
    the stem conv is VALID (no padding).
  * squeeze-and-excitation: mean-pool -> Dense -> swish -> Dense -> sigmoid -> scale.
  * residual (with stochastic depth) on every block except the first of each stage.
  * frontend: 'SAME' framing (160 samples of zero padding each side for a 5 s clip), symmetric Hann window divided by
    its sum, |rfft| with nfft 1024 (magnitude, not power), the TF / HTK mel matrix (128 bins, 60 Hz - 16 kHz),
    0.1 * log(max(mel, 1e-5)). (The paper's "uncentered frames" does not match the released model.)

Layout: the backbone works on (batch, 1, time=500, freq=128) spectrograms, which is the orientation Perch was trained
on. The preprocessor's librosa mels are (freq, time), so `PerchV2Backbone(input_layout="freq_time")` (the default)
transposes on the way in. Mel bin 0 is the lowest frequency in both.
"""
import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 32_000
WINDOW_SAMPLES = 160_000  # 5 s
EMBEDDING_DIM = 1536

# EfficientNet-B0 stages (num_blocks, features, kernel, stride, expand_ratio), scaled to B3 as in chirp/models/efficientnet.py
_B0_STAGES = [(1, 16, 3, 1, 1), (2, 24, 3, 2, 6), (2, 40, 5, 2, 6), (3, 80, 3, 2, 6), (3, 112, 5, 1, 6),
              (4, 192, 5, 2, 6), (1, 320, 3, 1, 6)]
_B3_WIDTH, _B3_DEPTH = 1.2, 1.4
_STEM_FEATURES, _HEAD_FEATURES, _REDUCTION_RATIO = 32, 1280, 4


def _round_features(features: int, width: float = _B3_WIDTH, divisor: int = 8) -> int:
    scaled = features * width
    new = max(divisor, int(scaled + divisor / 2) // divisor * divisor)
    return int(new + divisor if new < 0.9 * scaled else new)


def _b3_blocks() -> list[tuple[int, int, int, int, int, bool]]:
    """(in_features, out_features, kernel, stride, expand_ratio, residual) of the 26 MBConv blocks."""
    blocks, in_features = [], _round_features(_STEM_FEATURES)
    for num_blocks, features, kernel, stride, expand in _B0_STAGES:
        out_features = _round_features(features)
        for i in range(int(math.ceil(_B3_DEPTH * num_blocks))):
            blocks.append((in_features, out_features, kernel, stride if i == 0 else 1, expand, i > 0))
            in_features = out_features
    return blocks


def _batch_norm(features: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(features, eps=1e-5, momentum=0.01)


class _SqueezeExcitation(nn.Module):
    def __init__(self, features: int, reduced: int) -> None:
        super().__init__()
        self.reduce = nn.Linear(features, reduced)
        self.expand = nn.Linear(reduced, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(self.expand(F.silu(self.reduce(x.mean(dim=(2, 3))))))
        return x * s[:, :, None, None]


class _MBConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        mid = in_features * expand_ratio
        self.kernel, self.stride = kernel, stride
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_features, mid, 1, bias=False)
            self.expand_bn = _batch_norm(mid)
        else:
            self.expand_conv = self.expand_bn = None
        self.depthwise_conv = nn.Conv2d(mid, mid, kernel, stride=stride, padding=0 if stride == 2 else kernel // 2,
                                        groups=mid, bias=False)
        self.depthwise_bn = _batch_norm(mid)
        self.se = _SqueezeExcitation(mid, int(mid // (_REDUCTION_RATIO * expand_ratio)))
        self.project_conv = nn.Conv2d(mid, out_features, 1, bias=False)
        self.project_bn = _batch_norm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.expand_conv is not None:
            x = F.silu(self.expand_bn(self.expand_conv(x)))
        if self.stride == 2:  # chirp's _pad_width: (k//2 - (1 - n % 2), k//2) on each spatial dim
            k = self.kernel // 2
            h, w = x.shape[-2:]
            x = F.pad(x, (k - 1 + w % 2, k, k - 1 + h % 2, k))
        x = F.silu(self.depthwise_bn(self.depthwise_conv(x)))
        x = self.se(x)
        return self.project_bn(self.project_conv(x))


class _ConvBNSiLU(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel, stride=stride, padding=0, bias=False)
        self.bn = _batch_norm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.bn(self.conv(x)))


class PerchV2Backbone(nn.Module):
    """Perch v2's EfficientNet-B3 embedding network: log-mel spectrogram -> 1536-d embedding (mean over the 16 x 4
    spatial grid, as the SavedModel's `embedding` output). Load weights with `load_perch_v2_backbone`.

    input_layout: "freq_time" for (B, 1, 128 mels, 500 frames), the preprocessor's orientation (transposed here), or
        "time_freq" for (B, 1, 500, 128), Perch's own. A (B, H, W) input is treated as (B, 1, H, W).
    drop_path_rate: stochastic depth on the residual blocks, train mode only. Perch was trained with 0.2 (chirp's
        survival_probability 0.8); 0 disables it. Eval output is unaffected either way.
    input_scale: multiplies the input first, to map other log units onto Perch's 0.1 * ln(mel amplitude)
        (0.1 * ln(10) / 20 for amplitude dB). 1.0 for PerchV2Frontend's output.
    freeze_batchnorm: keep every BatchNorm in eval mode (running statistics) even in train(); affine weights still
        train.
    """

    out_features = EMBEDDING_DIM

    def __init__(self, input_layout: Literal["freq_time", "time_freq"] = "freq_time", drop_path_rate: float = 0.0,
                 input_scale: float = 1.0, freeze_batchnorm: bool = False) -> None:
        super().__init__()
        if input_layout not in ("freq_time", "time_freq"):
            raise ValueError(f"input_layout must be 'freq_time' or 'time_freq', got {input_layout!r}")
        self.input_layout = input_layout
        self.drop_path_rate = drop_path_rate
        self.input_scale = input_scale
        self.freeze_batchnorm = freeze_batchnorm
        blocks = _b3_blocks()
        self.stem = _ConvBNSiLU(1, blocks[0][0], 3, 2)
        self.blocks = nn.ModuleList(_MBConv(i, o, k, s, e) for i, o, k, s, e, _ in blocks)
        self.residual = [r for *_, r in blocks]
        self.head = _ConvBNSiLU(blocks[-1][1], _round_features(_HEAD_FEATURES), 1, 1)

    def _drop_path(self, y: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path_rate == 0.0:
            return y
        keep = 1.0 - self.drop_path_rate
        mask = torch.rand(y.shape[0], 1, 1, 1, device=y.device, dtype=y.dtype) < keep
        return y * mask / keep

    def train(self, mode: bool = True) -> "PerchV2Backbone":
        super().train(mode)
        if mode and self.freeze_batchnorm:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def spatial(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1536, 16, 4) feature map (time x freq) = the SavedModel's `spatial_embedding`, channels first."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if self.input_layout == "freq_time":
            x = x.transpose(-1, -2)
        if self.input_scale != 1.0:
            x = x * self.input_scale
        x = self.stem(x)
        for block, residual in zip(self.blocks, self.residual):
            y = block(x)
            x = x + self._drop_path(y) if residual else y
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(x).mean(dim=(2, 3))


def _hz_to_mel(f):
    return 1127.0 * np.log1p(np.asarray(f, dtype=np.float64) / 700.0)


def _mel_matrix(num_mel_bins: int, num_spectrogram_bins: int, sample_rate: int, fmin: float, fmax: float) -> np.ndarray:
    """tf.signal.linear_to_mel_weight_matrix / chirp.signal's port: (num_spectrogram_bins, num_mel_bins), HTK mel."""
    linear = _hz_to_mel(np.linspace(0.0, sample_rate / 2, num_spectrogram_bins)[1:])[:, None]
    edges = _hz_to_mel(np.array([fmin, fmax]))
    edges = np.linspace(edges[0], edges[1], num_mel_bins + 2)
    lower, center, upper = edges[None, :-2], edges[None, 1:-1], edges[None, 2:]
    weights = np.maximum(0.0, np.minimum((linear - lower) / (center - lower), (upper - linear) / (upper - center)))
    return np.pad(weights, ((1, 0), (0, 0)))


class PerchV2Frontend(nn.Module):
    """Perch v2's log-mel frontend: (B, samples) at 32 kHz -> (B, 1, frames, 128) for "time_freq" (Perch's own, the
    SavedModel's `spectrogram` output with a channel axis) or (B, 1, 128, frames) for "freq_time". No parameters;
    5 s in -> 500 frames. Only needed for raw-audio input; anything producing the same values can replace it."""

    def __init__(self, output_layout: Literal["freq_time", "time_freq"] = "freq_time") -> None:
        super().__init__()
        self.output_layout = output_layout
        self.hop, self.win, self.nfft = 320, 640, 1024
        window = np.hanning(self.win)  # symmetric Hann, as jnp.hanning in chirp's SimpleMelspec
        self.register_buffer("window", torch.tensor(window / window.sum(), dtype=torch.float32), persistent=False)
        mel = _mel_matrix(128, self.nfft // 2 + 1, SAMPLE_RATE, 60.0, 16_000.0)
        self.register_buffer("mel_matrix", torch.tensor(mel, dtype=torch.float32), persistent=False)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        n_frames = -(-audio.shape[-1] // self.hop)
        pad = max((n_frames - 1) * self.hop + self.win - audio.shape[-1], 0)
        frames = F.pad(audio, (pad // 2, pad - pad // 2)).unfold(-1, self.win, self.hop)[..., :n_frames, :]
        magnitude = torch.fft.rfft(frames * self.window, n=self.nfft).abs()
        out = 0.1 * torch.log(torch.clamp(magnitude @ self.mel_matrix, min=1e-5))  # (B, frames, 128)
        out = out.unsqueeze(1)
        return out.transpose(-1, -2) if self.output_layout == "freq_time" else out


def perch_v2_state_dict(variables: dict[str, np.ndarray], model: PerchV2Backbone | None = None) -> dict[str, torch.Tensor]:
    """Map the SavedModel's variables (by name, e.g. 'params.embedding_model.backbone.MBConv_3.ExpandConv.kernel',
    'batch_stats.embedding_model.backbone.Stem_0.BatchNorm_0.mean') onto a PerchV2Backbone state dict. Raises if
    any backbone variable is unused or any model tensor is left unfilled / mis-shaped. Non-backbone variables (the
    14,795-class ProtoPNet head) are ignored."""
    model = model or PerchV2Backbone()
    prefix = ".embedding_model.backbone."
    flax = {f"{k.split(prefix)[0]}.{k.split(prefix)[1]}": v for k, v in variables.items() if prefix in k}

    def conv(name: str) -> torch.Tensor:  # flax (kh, kw, in, out) -> torch (out, in, kh, kw)
        return torch.from_numpy(np.ascontiguousarray(np.asarray(flax.pop(f"params.{name}.kernel")).transpose(3, 2, 0, 1)))

    def bn(torch_name: str, flax_name: str) -> dict[str, torch.Tensor]:
        get = lambda k: torch.from_numpy(np.asarray(flax.pop(k)).copy())  # noqa: E731
        return {f"{torch_name}.weight": get(f"params.{flax_name}.scale"), f"{torch_name}.bias": get(f"params.{flax_name}.bias"),
                f"{torch_name}.running_mean": get(f"batch_stats.{flax_name}.mean"),
                f"{torch_name}.running_var": get(f"batch_stats.{flax_name}.var"),
                f"{torch_name}.num_batches_tracked": torch.tensor(0)}

    def dense(torch_name: str, flax_name: str) -> dict[str, torch.Tensor]:  # flax (in, out) -> torch (out, in)
        return {f"{torch_name}.weight": torch.from_numpy(np.ascontiguousarray(np.asarray(flax.pop(f"params.{flax_name}.kernel")).T)),
                f"{torch_name}.bias": torch.from_numpy(np.asarray(flax.pop(f"params.{flax_name}.bias")).copy())}

    state = {"stem.conv.weight": conv("Stem_0.Conv_0"), **bn("stem.bn", "Stem_0.BatchNorm_0"),
             "head.conv.weight": conv("Head_0.Conv_0"), **bn("head.bn", "Head_0.BatchNorm_0")}
    for i, block in enumerate(model.blocks):
        f, t = f"MBConv_{i}", f"blocks.{i}"
        if block.expand_conv is not None:
            state[f"{t}.expand_conv.weight"] = conv(f"{f}.ExpandConv")
            state.update(bn(f"{t}.expand_bn", f"{f}.ExpandBatchNorm"))
        state[f"{t}.depthwise_conv.weight"] = conv(f"{f}.DepthwiseConv")
        state.update(bn(f"{t}.depthwise_bn", f"{f}.DepthwiseBatchNorm"))
        state.update(dense(f"{t}.se.reduce", f"{f}.SqueezeAndExcitation_0.Reduce"))
        state.update(dense(f"{t}.se.expand", f"{f}.SqueezeAndExcitation_0.Expand"))
        state[f"{t}.project_conv.weight"] = conv(f"{f}.ProjectConv")
        state.update(bn(f"{t}.project_bn", f"{f}.ProjectBatchNorm"))

    if flax:
        raise ValueError(f"{len(flax)} backbone variables not mapped, e.g. {sorted(flax)[:5]}")
    expected = model.state_dict()
    missing, extra = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    bad_shapes = [k for k in set(expected) & set(state) if expected[k].shape != state[k].shape]
    if missing or extra or bad_shapes:
        raise ValueError(f"state dict mismatch: missing {missing[:5]}, extra {extra[:5]}, shapes {bad_shapes[:5]}")
    return {k: v.float() if v.is_floating_point() else v for k, v in state.items()}


def load_perch_v2_backbone(checkpoint_path: str, input_layout: Literal["freq_time", "time_freq"] = "freq_time",
                           drop_path_rate: float = 0.0, input_scale: float = 1.0,
                           freeze_batchnorm: bool = False) -> PerchV2Backbone:
    """A PerchV2Backbone with the weights written by scripts/perch/convert_perch_v2_to_torch.py."""
    model = PerchV2Backbone(input_layout=input_layout, drop_path_rate=drop_path_rate, input_scale=input_scale,
                            freeze_batchnorm=freeze_batchnorm)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return model
