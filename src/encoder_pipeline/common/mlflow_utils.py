from typing import Optional

import mlflow
from pathlib import Path

from encoder_pipeline.common.config_utils import PipelineConfig

def configure_mlflow(config: PipelineConfig) -> None:
    """Sets the MLflow tracking URI (if given) and active experiment."""
    if config.mlflow_tracking_uri is not None:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)


def configure_datasets_mlflow(experiment_name: str, mlflow_tracking_uri: Optional[str] = None) -> None:
    """Sets the MLflow tracking URI (if given) and active experiment, for
    dataset-generation scripts (e.g. build_dclde_2027_annotations.py) that
    aren't driven by a full PipelineConfig. Callers own their own experiment
    path, e.g. "Datasets/DCLDE_2027" -- MLflow renders '/' in experiment names
    as a folder tree -- separate from training experiments."""
    if mlflow_tracking_uri is not None:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)


def flatten_params(prefix: str, obj: dict) -> dict:
    """Flattens a nested dict into dotted-key: value pairs, for mlflow.log_params."""
    flat = {}
    for key, value in obj.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_params(full_key, value))
        else:
            flat[full_key] = value
    return flat

def download_artifact(root_path: str, run_id: str, artifact_path: str | None = None) -> Path:
    """Downloads an artifact from mlflow if not already cached locally."""
    local_dir = Path(root_path) / "artifact_cache" / run_id
    local_path = local_dir / artifact_path if artifact_path else local_dir
    if local_path.exists():
        return local_path
    return Path(mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path, dst_path=str(local_dir),
    ))