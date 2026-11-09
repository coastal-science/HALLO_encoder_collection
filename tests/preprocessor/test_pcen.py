"""PCEN on float-scale audio must not suppress the start of the window (librosa's default filter start is 1.0)."""

from types import SimpleNamespace

import numpy as np

from encoder_pipeline.preprocessor.config import SpectrogramConfig
from encoder_pipeline.preprocessor.spectrogram import Spectrogram


def _pcen_of_stationary_noise(time_constant: float) -> np.ndarray:
    sr, seconds = 250, 30
    audio = np.random.default_rng(0).normal(scale=0.01, size=sr * seconds).astype(np.float32)
    config = SpectrogramConfig.model_validate({
        "stft": {"n_fft": 128, "hop_length": 32}, "freq_scale": "mag", "mag": {"power": 2.0},
        "dynamic": "pcen", "pcen": {"time_constant": time_constant},
    })
    annotation = SimpleNamespace(audio=audio, file=SimpleNamespace(sr=sr))
    spec = Spectrogram(config)
    return spec.apply_dynamic(annotation, spec.compute_magnitude(annotation))


def test_pcen_has_no_startup_suppression_on_stationary_noise():
    for time_constant in (0.4, 2.0, 10.0):
        out = _pcen_of_stationary_noise(time_constant)
        frames_per_s = out.shape[1] / 30
        first = out[:, : int(2 * frames_per_s)].mean()
        last = out[:, -int(5 * frames_per_s):].mean()
        assert 0.8 < first / last < 1.25, f"time_constant={time_constant}: first 2 s / last 5 s = {first / last:.3f}"
