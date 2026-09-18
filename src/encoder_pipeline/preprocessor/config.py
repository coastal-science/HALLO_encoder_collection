from typing import Literal, Optional

from encoder_pipeline.common.base import StrictBaseModel
from encoder_pipeline.common.file_utils import resolve_cpu_workers


class AudioFileConfig(StrictBaseModel):

    resample_sr: Optional[int] = None
    """Sample rate every file is resampled to on load. None keeps each
    file's native rate"""


class AnnotationConfig(StrictBaseModel):

    time_offset: float = 2.5
    """Half-width, in seconds, of the padded window kept around the
    annotation's center."""


class STFTConfig(StrictBaseModel):
    """Mirrors librosa.stft's constructor field-for-field."""

    n_fft: int = 2048
    """Number of samples i.e. FFT length; also the window length if win_length is None."""

    hop_length: Optional[int] = None
    """How many samples to move the window by i.e. samples between successive frames. Defaults to n_fft // 4 if None."""

    win_length: Optional[int] = None
    """Window length in samples. Defaults to n_fft if None."""

    window: str = "hann"
    """Window function name -- see scipy.signal.get_window for accepted specs."""

    center: bool = True
    """If True, zero-pads by n_fft // 2 on each side so frame t is centered
    at y[t * hop_length]. If False, frame t starts at y[t * hop_length] with
    no padding."""

    pad_mode: str = "constant"
    """Padding mode used when center=True; 'constant' zero-pads."""

    def stft_kwargs(self) -> dict:
        return self.model_dump()


class MagConfig(StrictBaseModel):
    """Linear-frequency spectrogram, no mel filterbank."""

    power: float = 1.0
    """1.0 for amplitude, 2.0 for power."""


class MelConfig(StrictBaseModel):

    """Mirrors librosa.filters.mel's constructor"""

    n_mels: int = 128
    """Number of mel bands to generate. """

    fmin: float = 0.0
    """Lowest frequency, in Hz."""

    fmax: Optional[float] = None
    """Highest frequency, in Hz. Defaults to sr / 2 if None."""

    htk: bool = False
    """Use the HTK formula instead of Slaney's."""

    norm: Optional[str] = "slaney"
    """Filter normalization; 'slaney' divides each filter by its bandwidth."""

    power: float = 2.0
    """1.0 for amplitude, 2.0 for power."""

    def mel_kwargs(self) -> dict:
        return self.model_dump(exclude={"power"})


class DbConfig(StrictBaseModel):
    """Mirrors librosa.power_to_db / amplitude_to_db's constructor"""

    ref: float = 1.0
    amin: float = 1e-10
    top_db: Optional[float] = 80.0


class PcenConfig(StrictBaseModel):
    """Mirrors librosa.pcen's constructor"""

    gain: float = 0.98
    bias: float = 2.0
    power: float = 0.5
    time_constant: float = 0.4
    eps: float = 1e-6


class SpectrogramConfig(StrictBaseModel):

    stft: STFTConfig = STFTConfig()

    freq_scale: Literal["mag", "mel"] = "mag"
    mag: MagConfig = MagConfig()
    mel: MelConfig = MelConfig()

    dynamic: Literal["db", "pcen"] = "db"
    db: DbConfig = DbConfig()
    pcen: PcenConfig = PcenConfig()


class DatasetConfig(StrictBaseModel):

    annotations_csv: Optional[str] = None
    """Path to the annotations CSV. Exactly one of this or annotations_mlflow_id."""
    annotations_mlflow_id: Optional[str] = None
    """MLflow run_id to resolve annotations_csv from (its 'annotations_path' param). Exactly one of this or annotations_csv."""
    max_workers: Optional[int] = None
    """None or a positive int, same as ProcessPoolExecutor; also accepts
    joblib-style negative values (-1 = all cores, -2 = all but one, ...)."""
    metadata_columns: Optional[list[str]] = None # specify metadata columns which should be included.
    local_file_col: Optional[str] = "LocalPath" # column in annotations csv where local audio file paths are specified
    uid_col: Optional[str] = "uid" # column in annotations csv whuich specifies the row-level uid
    classes_to_drop: Optional[list[str]] = None
    """Label values (matched against the 'Labels' column) to drop from the
    annotations before building the dataset. Their rows never enter the HDF5,
    so they're absent from every later split."""
    def resolve_max_workers(self) -> Optional[int]:
        if self.max_workers is None:
            return None
        return resolve_cpu_workers(self.max_workers, floor=1)


class PreprocessorConfig(StrictBaseModel):
    spectrogram: SpectrogramConfig = SpectrogramConfig()
    audio_file: AudioFileConfig = AudioFileConfig()
    annotation: AnnotationConfig = AnnotationConfig()
    dataset: DatasetConfig
    run_name: Optional[str] = None # MLflow run name. If unset, MLflow auto-generates one.
