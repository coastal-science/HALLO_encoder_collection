from typing import Literal, Optional

from encoder_pipeline.common.base import StrictBaseModel


class PerchConfig(StrictBaseModel):
    """perch_hoplite zoo model + raw-audio embedding params, used when
    EmbeddingsConfig.source is 'perch_hoplite'."""

    preset: str = "perch_v2"
    """perch_hoplite preset name, e.g. 'perch_v2', 'perch_8', 'surfperch'."""
    batch_size: int = 64
    load_workers: int = 8


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
    reuse_embeddings: bool = False
    """Skip the encoder forward pass and load each fold's embeddings from a
    prior run's data_dir/embeddings/<run_id>/fold<n>.h5, so only the linear
    probe reruns. Errors if no fold file is found. The preprocessor dataset and
    checkpoints are not read, so reuse_classes must be given."""
    reuse_classes: Optional[list[str]] = None
    """Class list in saved-label index order, required with reuse_embeddings
    (no dataloaders are built to supply it). Normally sorted(set(...)) of the
    training labels after model_trainer.dataloader.class_label_map."""
    linear_probe_epochs: int = 1000
    linear_probe_lr: float = 3e-4
    linear_probe_label_map: Optional[dict[str, str]] = None
    """Collapses labels for the probe the same way
    DataLoaderConfig.class_label_map does in training, e.g. {"SRKW": "KW",
    "TKW": "KW"}. Applied on top of the training map (labels are already
    collapsed by it); no rows are dropped. null = no extra collapsing."""
