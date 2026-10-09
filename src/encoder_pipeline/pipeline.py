import argparse
from contextlib import nullcontext
from pathlib import Path

import mlflow
from torch.utils.data import DataLoader

from encoder_pipeline.common.config_utils import PipelineConfig, load_pipeline_config
from encoder_pipeline.common.mlflow_utils import configure_mlflow, flatten_params
from encoder_pipeline.embeddings.runner import generate_embeddings, rerun_linear_probe
from encoder_pipeline.model_trainer.data_loader import build_dataloaders
from encoder_pipeline.model_trainer.runner import run_model_trainer
from encoder_pipeline.preprocessor.runner import run_preprocessor


def open_untrained_run(config: PipelineConfig, dataset_path: str) -> tuple[str, list[dict[str, DataLoader]]]:
    """A model_trainer-style run (nested under the dataset's preprocessor run)
    with split dataloaders but no training, for sources that need no checkpoint."""
    matches = mlflow.search_runs(
        filter_string=f"params.dataset_path = '{dataset_path}'", order_by=["start_time ASC"], max_results=1,
    )
    parent_run_id = matches.iloc[0]["run_id"] if not matches.empty else None
    with mlflow.start_run(run_id=parent_run_id) if parent_run_id else nullcontext():
        with mlflow.start_run(nested=parent_run_id is not None, run_name=config.model_trainer.run_name) as run:
            mlflow.log_params(flatten_params("model_trainer", config.model_trainer.model_dump()))
            mlflow.log_param("dataset_path", dataset_path)
            dataloaders = build_dataloaders(dataset_path, config.model_trainer.dataloader, config.data_dir)
    return run.info.run_id, dataloaders


def run_pipeline(config: PipelineConfig) -> None:
    configure_mlflow(config)
    if config.embeddings.reuse_embeddings_path:
        rerun_linear_probe(config.embeddings, config.model_trainer.dataloader.class_label_map)
        return

    is_perch = config.embeddings.source == "perch_hoplite"
    if config.embeddings.mlflow_id is None and (is_perch or config.embeddings.checkpoint_mlflow_id):
        dataset_path = run_preprocessor(config.preprocessor, config.data_dir, metadata_only=is_perch)
        run_id, dataloaders = open_untrained_run(config, dataset_path)
    elif config.embeddings.mlflow_id is None:
        dataset_path = run_preprocessor(config.preprocessor, config.data_dir)
        run_id, dataloaders = run_model_trainer(config.model_trainer, dataset_path, config.data_dir)
        if config.model_trainer.tune is not None and config.model_trainer.tune.enabled:
            print("model_trainer.tune enabled, skipping embeddings .")
            return
    else:
        run_id = config.embeddings.mlflow_id
        dataset_path = mlflow.get_run(run_id).data.params["dataset_path"]
        with mlflow.start_run(run_id=run_id):
            dataloaders = build_dataloaders(dataset_path, config.model_trainer.dataloader, config.data_dir)

    generate_embeddings(
        config.embeddings, dataloaders, run_id, config.data_dir, config.model_trainer.dataloader.class_label_map,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Base config, e.g. configs/base.yaml.")
    parser.add_argument("--override", type=Path, default=None, help="Override config, deep-merged onto --config.")
    args = parser.parse_args()
    config = load_pipeline_config(args.config, args.override)
    run_pipeline(config)


if __name__ == "__main__":
    main()
