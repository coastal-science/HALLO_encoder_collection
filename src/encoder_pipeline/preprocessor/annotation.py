from typing import Optional

import librosa
import numpy as np

from encoder_pipeline.preprocessor.config import AnnotationConfig


class AudioFile:
    """Decodes a file once so multiple Annotations on it can share the audio
    instead of each re-reading/re-decoding the same file."""

    def __init__(self, file_path: str, resample_sr: Optional[int] = None) -> None:
        self.file_path: str = file_path
        self.audio, self.sr = librosa.load(file_path, sr=resample_sr)

    def slice(self, time_start: float, duration: float) -> np.ndarray:
        start_sample = int(time_start * self.sr)
        end_sample = start_sample + int(duration * self.sr)
        return self.audio[start_sample:end_sample]


class Annotation:
    def __init__(self, file: AudioFile, label: str, time_start: float, duration: float, config: AnnotationConfig) -> None:
        self.file = file
        self.label = label
        self.time_offset = config.time_offset
        self.duration: float = 2 * config.time_offset
        file_duration = len(file.audio) / file.sr
        Annotation.check_window(file_duration, config)
        # window is centered on the annotation; annotations longer than it are center-cropped
        center = time_start + duration / 2
        self.time_start: float = max(0.0, min(center - config.time_offset, file_duration - self.duration))

    @staticmethod
    def check_window(file_duration: float, config: AnnotationConfig) -> None:
        """Raises ValueError if the 2 * time_offset window is longer than the file."""
        window = 2 * config.time_offset
        if window > file_duration:
            raise ValueError(
                f"time_offset={config.time_offset} gives a padded window of {window}s, "
                f"longer than the file itself ({file_duration}s)."
            )

    @property
    def audio(self) -> np.ndarray:
        return self.file.slice(self.time_start, self.duration)

