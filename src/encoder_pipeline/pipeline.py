import argparse
from pathlib import Path

import mlflow

from encoder_pipeline.common.config_utils import PipelineConfig, load_pipeline_config
from encoder_pipeline.common.mlflow_utils import configure_mlflow
from encoder_pipeline.embeddings.runner import generate_embeddings
from encoder_pipeline.model_trainer.data_loader import build_dataloaders
from encoder_pipeline.model_trainer.runner import run_model_trainer
from encoder_pipeline.preprocessor.runner import run_preprocessor


def run_pipeline(config: PipelineConfig) -> None:
    configure_mlflow(config)

    if config.embeddings.mlflow_id is None:
        dataset_path = run_preprocessor(config.preprocessor, config.data_dir)
        run_id, dataloaders = run_model_trainer(config.model_trainer, dataset_path, config.data_dir)
    elif config.embeddings.reuse_embeddings:
        run_id = config.embeddings.mlflow_id
        dataloaders = None
    else:
        run_id = config.embeddings.mlflow_id
        dataset_path = mlflow.get_run(run_id).data.params["dataset_path"]
        with mlflow.start_run(run_id=run_id):
            dataloaders = build_dataloaders(dataset_path, config.model_trainer.dataloader, config.data_dir)

    generate_embeddings(config.embeddings, dataloaders, run_id, config.data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Base config, e.g. configs/base.yaml.")
    parser.add_argument("--override", type=Path, default=None, help="Override config, deep-merged onto --config.")
    args = parser.parse_args()
    config = load_pipeline_config(args.config, args.override)
    run_pipeline(config)


if __name__ == "__main__":
    main()
