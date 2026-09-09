import argparse
import pickle
from pathlib import Path
from typing import Optional

import mlflow
from torch.utils.data import DataLoader

from encoder_pipeline.common.config_utils import load_pipeline_config
from encoder_pipeline.common.mlflow_utils import configure_mlflow, download_artifact, flatten_params
from encoder_pipeline.model_trainer.config import ModelTrainerConfig
from encoder_pipeline.model_trainer.data_loader import build_dataloaders
from encoder_pipeline.model_trainer.hpo import run_tuning
from encoder_pipeline.model_trainer.train import train_model


def run_model_trainer(
    config: ModelTrainerConfig, dataset_path: str, data_dir: str,
) -> tuple[str, Optional[list[dict[str, DataLoader]]]]:
    # search for dataset on mlflow
    matches = mlflow.search_runs(
        filter_string=f"params.dataset_path = '{dataset_path}'", order_by=["start_time ASC"], max_results=1,
    )
    parent_run_id = matches.iloc[0]["run_id"] if not matches.empty else None

    spectrogram_config = None
    if parent_run_id is not None:
        artifact_path = download_artifact(data_dir, parent_run_id, artifact_path="spectrogram_config.pkl")
        with open(artifact_path, "rb") as f:
            spectrogram_config = pickle.load(f)
    else:
        print(f"no preprocessor run found for dataset_path={dataset_path} -- training without a spectrogram_config")

    def _train() -> tuple[str, Optional[list[dict[str, DataLoader]]]]:
        with mlflow.start_run(nested=parent_run_id is not None, run_name=config.run_name) as run:
            mlflow.log_params(flatten_params("model_trainer", config.model_dump()))
            mlflow.log_param("dataset_path", dataset_path)
            if config.tune is not None and config.tune.enabled:
                run_tuning(
                    config, dataset_path, data_dir, run.info.run_id,
                    mlflow.get_tracking_uri(), mlflow.get_experiment(run.info.experiment_id).name,
                    spectrogram_config,
                )
                return run.info.run_id, None
            dataloaders = build_dataloaders(dataset_path, config.dataloader, data_dir)
            train_model(config, dataloaders, data_dir, spectrogram_config)
            return run.info.run_id, dataloaders

    if parent_run_id is not None:
        with mlflow.start_run(run_id=parent_run_id):
            return _train()
    return _train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Base config, e.g. configs/base.yaml.")
    parser.add_argument("--override", type=Path, default=None, help="Override config, deep-merged onto --config.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset produced by preprocessor.")
    args = parser.parse_args()
    pipeline_config = load_pipeline_config(args.config, args.override)

    configure_mlflow(pipeline_config)
    run_model_trainer(pipeline_config.model_trainer, args.dataset_path, pipeline_config.data_dir)


if __name__ == "__main__":
    main()
