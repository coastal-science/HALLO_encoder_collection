import librosa
import numpy as np
import scipy.signal

from encoder_pipeline.preprocessor.annotation import Annotation
from encoder_pipeline.preprocessor.config import SpectrogramConfig

class Spectrogram:
    """Class for generating spectrograms from raw audio files
    """

    def __init__(self, config: SpectrogramConfig):
        self.config = config


    def compute_magnitude(self, annotation: Annotation) -> np.ndarray:
        """Real-valued magnitude (config.mag.power=1) or power
        (config.mag.power=2) spectrogram, derived from compute()'s complex
        STFT via librosa.magphase"""
        raw_stft = librosa.stft(annotation.audio, **self.config.stft.stft_kwargs())
        magnitude, _phase = librosa.magphase(raw_stft, power=self.config.mag.power)
        return magnitude

    def compute_mel(self, annotation: Annotation) -> np.ndarray:
        """Computes the mel spectrogram"""
        return librosa.feature.melspectrogram(
            y=annotation.audio, sr=annotation.file.sr, power=self.config.mel.power,
            **self.config.stft.stft_kwargs(), **self.config.mel.mel_kwargs(),
        )

    def apply_dynamic(self, annotation: Annotation, spec: np.ndarray) -> np.ndarray:
        """Applies whichever dynamic-range compression config.dynamic
        selects to an already-computed spectrogram.

        db: librosa.power_to_db or amplitude_to_db, whichever matches the
        active freq_scale's power convention (config.mag.power /
        config.mel.power: 1.0 = amplitude, 2.0 = power).

        pcen: librosa.pcen (Per-Channel Energy Normalization). Expects a
        power spectrogram (config.mel.power=2.0) -- PCEN's own AGC assumes
        energy, not amplitude. The smoother is started at steady state on
        each band's mean over the first time_constant seconds: librosa's
        default start (steady state for an input of 1.0) is far above
        float-audio energies (~1e-6), so the background estimate took ~10x
        time_constant to decay and suppressed the start of every window. A
        single first frame is too noisy (and half zero-padding when center=True).
        """
        if self.config.dynamic == "pcen":
            pcen_config = self.config.pcen
            sr = annotation.file.sr
            hop_length = self.config.stft.hop_length or self.config.stft.n_fft // 4
            # same smoothing coefficient librosa.pcen derives from time_constant
            t_frames = pcen_config.time_constant * sr / float(hop_length)
            b = (np.sqrt(1 + 4 * t_frames**2) - 1) / (2 * t_frames**2)
            n_init = int(np.clip(round(t_frames), 1, spec.shape[-1]))
            zi = scipy.signal.lfilter_zi([b], [1, b - 1]) * spec[..., :n_init].mean(axis=-1, keepdims=True)
            return librosa.pcen(
                spec,
                sr=sr,
                hop_length=hop_length,
                gain=pcen_config.gain,
                bias=pcen_config.bias,
                power=pcen_config.power,
                time_constant=pcen_config.time_constant,
                eps=pcen_config.eps,
                zi=zi,
            )
        elif self.config.dynamic == "db":
            db_config = self.config.db
            active_power = self.config.mel.power if self.config.freq_scale == "mel" else self.config.mag.power
            to_db = librosa.power_to_db if active_power == 2.0 else librosa.amplitude_to_db
            return to_db(spec, ref=db_config.ref, amin=db_config.amin, top_db=db_config.top_db)
        else:
            raise ValueError(f"self.config.dynamic must either be 'db' or 'pcen', {self.config.dynamic} was passed")