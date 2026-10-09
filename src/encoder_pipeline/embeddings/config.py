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


class EmbeddingsConfig(StrictBaseModel):
    source: Literal["HALLO_encoder_collection", "perch_hoplite"] = "HALLO_encoder_collection"
    """"HALLO_encoder_collection": run_id's model_trainer checkpoint, from
    local disk if present, else downloaded from the MLflow tracking server.
    "perch_hoplite": a pretrained perch zoo model (see perch below); re-reads
    raw audio from the dataset's LocalPath/FileBeginSec/Duration columns and
    ignores the spectrogram pipeline. No checkpoint needed -- run_id is just
    the run the embeddings are logged under."""
    perch: PerchConfig = PerchConfig()
    mlflow_id: Optional[str] = None
    """run_id to embed. null = the run this pipeline invocation's own
    model_trainer stage just produced -- see pipeline.run_pipeline."""
    checkpoint_mlflow_id: Optional[str] = None
    """run_id whose model_trainer checkpoint embeds this config's own
    preprocessor dataset, logged under a new untrained run. null = run_id's
    own checkpoint."""
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
