import argparse
import pickle
from pathlib import Path

import mlflow

from encoder_pipeline.common.config_utils import load_pipeline_config
from encoder_pipeline.common.mlflow_utils import configure_mlflow, flatten_params
from encoder_pipeline.preprocessor.config import DatasetConfig, PreprocessorConfig
from encoder_pipeline.preprocessor.dataset import Dataset
from encoder_pipeline.preprocessor.samples import save_sample_images


def resolve_annotations_csv(dataset_config: DatasetConfig) -> str:
    """Returns annotations_csv directly, or resolves it via annotations_mlflow_id's logged 'annotations_path' param."""
    if dataset_config.annotations_csv is not None and dataset_config.annotations_mlflow_id is not None:
        raise ValueError("set only one of dataset.annotations_csv / dataset.annotations_mlflow_id, not both")
    if dataset_config.annotations_mlflow_id is not None:
        return mlflow.get_run(dataset_config.annotations_mlflow_id).data.params["annotations_path"]
    if dataset_config.annotations_csv is not None:
        return dataset_config.annotations_csv
    raise ValueError("one of dataset.annotations_csv / dataset.annotations_mlflow_id must be set")


def run_preprocessor(config: PreprocessorConfig, data_dir: str, metadata_only: bool = False) -> str:
    """metadata_only: write only the annotation metadata (no spectrograms), for
    embedders that do their own audio preprocessing."""
    config.dataset.annotations_csv = resolve_annotations_csv(config.dataset)
    dataset = Dataset(
        config.spectrogram, config.audio_file, config.annotation, config.dataset, data_dir, config.run_name, metadata_only,
    )
    dataset.build_hdf5()
    # search if the hdf5 file has been logged to mlflow, by content hash (ignoring the timestamp)
    content_hash = Path(dataset.out_file).stem.rsplit("_", 1)[0]
    existing = mlflow.search_runs(
        filter_string=f"params.dataset_path LIKE '%{content_hash}%'", order_by=["start_time ASC"], max_results=1,
    )
    if not existing.empty:
        print(f"reusing run_id={existing.iloc[0]['run_id']} dataset_path={dataset.out_file}")
    else:
        with mlflow.start_run(run_name=config.run_name):
            mlflow.log_params(flatten_params("preprocessor", config.model_dump()))
            mlflow.log_param("dataset_path", dataset.out_file)
            mlflow.log_param("metadata_only", metadata_only)
            spec_config_path = Path(f"{data_dir}/preprocessor/{mlflow.active_run().info.run_id}/spectrogram_config.pkl")
            spec_config_path.parent.mkdir(parents=True, exist_ok=True)
            with spec_config_path.open("wb") as f:
                pickle.dump(config.spectrogram, f)
            mlflow.log_artifact(str(spec_config_path))
            if config.dataset.n_sample_images and not metadata_only:
                sample_paths = save_sample_images(
                    dataset.out_file, config, spec_config_path.parent / "sample_spectrograms", config.dataset.n_sample_images,
                )
                for sample_path in sample_paths:
                    mlflow.log_artifact(str(sample_path), artifact_path="sample_spectrograms")
            print(f"run_id={mlflow.active_run().info.run_id} dataset_path={dataset.out_file}")
    return dataset.out_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Base config, e.g. configs/base.yaml.")
    parser.add_argument("--override", type=Path, default=None, help="Override config, deep-merged onto --config.")
    args = parser.parse_args()
    pipeline_config = load_pipeline_config(args.config, args.override)

    configure_mlflow(pipeline_config)
    run_preprocessor(pipeline_config.preprocessor, pipeline_config.data_dir)


if __name__ == "__main__":
    main()
