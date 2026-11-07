import numpy as np

from encoder_pipeline.preprocessor.annotation import Annotation
from encoder_pipeline.preprocessor.config import AnnotationConfig


class _File:
    def __init__(self, audio, sr):
        self.audio, self.sr = audio, sr

    def slice(self, time_start, duration):
        start = int(time_start * self.sr)
        return self.audio[start:start + int(duration * self.sr)]


def _annotation(audio, peak_normalize):
    return Annotation(_File(audio, 100), "SRKW", 0.0, 2.0, AnnotationConfig(time_offset=1.0, peak_normalize=peak_normalize))


def test_peak_normalize_matches_perch_hoplite():
    rng = np.random.default_rng(0)
    audio = (0.01 * rng.standard_normal(200) + 0.003).astype(np.float32)
    out = _annotation(audio, 0.25).audio
    expected = audio - audio.mean()  # perch_hoplite zoo_interface.normalize_audio
    expected = expected / np.abs(expected).max() * 0.25
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-7)
    assert np.isclose(np.abs(out).max(), 0.25)


def test_peak_normalize_unset_returns_audio_unchanged():
    audio = np.linspace(-0.01, 0.02, 200).astype(np.float32)
    np.testing.assert_array_equal(_annotation(audio, None).audio, audio)


def test_peak_normalize_leaves_silence_alone():
    audio = np.zeros(200, dtype=np.float32)
    np.testing.assert_array_equal(_annotation(audio, 0.25).audio, audio)
