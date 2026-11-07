from typing import Literal, Optional

from encoder_pipeline.common.base import StrictBaseModel


class PerchConfig(StrictBaseModel):
    """perch_hoplite zoo model + raw-audio embedding params, used when
    EmbeddingsConfig.source is 'perch_hoplite'."""

    preset: str = "perch_v2"
    """perch_hoplite preset name, e.g. 'perch_v2', 'perch_8', 'surfperch'."""
    batch_size: int = 64
    load_workers: int = 8
    chunk_size: int = 4096
    """Unique clips loaded and embedded per chunk; bounds peak raw-audio memory."""


class WhisperConfig(StrictBaseModel):
    """Frozen OpenAI Whisper encoder (Hugging Face transformers) + raw-audio embedding params, used when
    EmbeddingsConfig.source is 'whisper'. Clips are re-read from raw audio at Whisper's 16 kHz (so nothing above
    8 kHz reaches it), turned into Whisper's own log-mel features, and run through the encoder with its 30 s
    positional table cut to the clip's length; embeddings are the mean over time of the chosen layer."""

    model_name: str = "openai/whisper-base"
    """Hugging Face checkpoint, e.g. openai/whisper-base, openai/whisper-small, openai/whisper-large-v3."""
    window_s: float = 5.0
    """Clip length fed to the encoder, centered on the annotation (Perch v2's 5 s, so the clips match perch_v2_probe's)."""
    layer: Optional[int] = None
    """Index into the encoder's hidden states (0 = after the conv stem + positions, i = after block i).
    null = the final output, after the encoder's last LayerNorm."""
    peak_normalize: Optional[float] = 0.25
    """Mean removed and absolute peak scaled to this per clip, before the features (as the Perch arms' 0.25), so
    hydrophone gain differences between deployments don't shift Whisper's fixed log-mel offset. null = raw levels."""
    batch_size: int = 64
    load_workers: int = 8
    chunk_size: int = 4096
    """Unique clips loaded and embedded per chunk; bounds peak raw-audio memory."""
    amp: bool = True
    """bf16 autocast for the encoder forward on cuda."""


class EmbeddingsConfig(StrictBaseModel):
    enabled: bool = True
    """false = stop after the model_trainer stage, e.g. for SSL pretraining on
    unlabelled data, where there is nothing to embed / probe."""
    source: Literal["HALLO_encoder_collection", "perch_hoplite", "whisper"] = "HALLO_encoder_collection"
    """"HALLO_encoder_collection": run_id's model_trainer checkpoint, from
    local disk if present, else downloaded from the MLflow tracking server.
    "perch_hoplite": a pretrained perch zoo model (see perch below); re-reads
    raw audio from the dataset's LocalPath/FileBeginSec/Duration columns and
    ignores the spectrogram pipeline. No checkpoint needed -- run_id is just
    the run the embeddings are logged under."""
    perch: PerchConfig = PerchConfig()
    whisper: WhisperConfig = WhisperConfig()
    """Used when source is 'whisper': like perch_hoplite, re-reads raw audio and ignores the spectrogram pipeline."""
    mlflow_id: Optional[str] = None
    """run_id to embed. null = the run this pipeline invocation's own
    model_trainer stage just produced -- see pipeline.run_pipeline."""
    checkpoint_mlflow_id: Optional[str] = None
    """run_id whose model_trainer checkpoint embeds this config's own
    preprocessor dataset, logged under a new untrained run. null = run_id's
    own checkpoint."""
    checkpoint_path: Optional[str] = None
    """A single model_trainer checkpoint file (e.g. a pretraining run's
    fold0_best.pt) whose frozen backbone embeds EVERY fold of this config's own
    preprocessor dataset, logged under a new untrained run -- a frozen-encoder
    probe. Takes precedence over checkpoint_mlflow_id. null = per-fold checkpoints."""
    reuse_embeddings_path: Optional[str] = None
    """A prior run's data_dir/embeddings/<run_id>/fold<n>.h5. Skips every other
    stage and only reruns the linear probe on it, as a new nested run under <run_id>."""
    linear_probe_epochs: int = 1000
    linear_probe_lr: float = 3e-4
    linear_probe_label_map: Optional[dict[str, str]] = None
    """Collapses labels for the probe the same way
    DataLoaderConfig.class_label_map does in training, e.g. {"SRKW": "KW",
    "TKW": "KW"}. Applied on top of the training map (labels are already
    collapsed by it); no rows are dropped. null = no extra collapsing."""
