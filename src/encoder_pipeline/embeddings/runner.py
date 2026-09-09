import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import mlflow
from torch.utils.data import DataLoader

from encoder_pipeline.common.config_utils import load_pipeline_config
from encoder_pipeline.common.mlflow_utils import configure_mlflow, flatten_params, download_artifact
from encoder_pipeline.embeddings.config import EmbeddingsConfig
from encoder_pipeline.embeddings.embed import HALLOEmbeddingModel, PerchEmbeddingModel
from encoder_pipeline.evaluation.linear_probe import LinearProbe, remap_labels
from encoder_pipeline.model_trainer.data_loader import build_dataloaders


def _perch_clip_frames(dataloaders: list[dict[str, DataLoader]]) -> list[dict[str, pd.DataFrame]]:
    """Per fold/split, the LocalPath/FileBeginSec/Duration/label rows PerchEmbeddingModel
    needs, pulled from the preprocessor HDF5 by each split's Subset indices."""
    base = next(iter(dataloaders[0].values())).dataset.dataset  # SpectrogramDataset
    with h5py.File(base.hdf5_path, "r") as h5:
        missing = [c for c in ("LocalPath", "FileBeginSec", "Duration") if c not in h5]
        if missing:
            raise ValueError(
                f"perch_hoplite source needs {missing} in the preprocessor HDF5 -- add them to "
                "preprocessor.dataset.metadata_columns and rebuild the dataset"
            )
        local_path, begin, duration = h5["LocalPath"].asstr()[:], h5["FileBeginSec"][:], h5["Duration"][:]
    labels = np.asarray(base.labels)
    return [
        {
            split: pd.DataFrame({
                "LocalPath": local_path[idx], "FileBeginSec": begin[idx],
                "Duration": duration[idx], "label": labels[idx],
            })
            for split, loader in loaders.items()
            for idx in [np.asarray(loader.dataset.indices)]
        }
        for loaders in dataloaders
    ]


def _load_fold_embeddings(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The {split: (embeddings, labels)} a prior run wrote to fold<n>.h5."""
    with h5py.File(path, "r") as h5:
        splits = sorted(k[: -len("_embeddings")] for k in h5 if k.endswith("_embeddings"))
        return {s: (h5[f"{s}_embeddings"][:], h5[f"{s}_labels"][:]) for s in splits}


def generate_embeddings(
    config: EmbeddingsConfig, dataloaders: list[dict[str, DataLoader]] | None, run_id: str, data_dir: str,
) -> None:
    """"""
    out_dir = Path(f"{data_dir}/embeddings/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    is_perch = config.source == "perch_hoplite"
    if config.reuse_embeddings:
        if not config.reuse_classes:
            raise ValueError("reuse_embeddings requires embeddings.reuse_classes")
        fold_loaders = [(fold, None) for fold, _ in enumerate(sorted(out_dir.glob("fold*.h5")))]
        if not fold_loaders:
            raise FileNotFoundError(f"reuse_embeddings: no fold*.h5 under {out_dir} -- run the stage once without it")
        checkpoint_desc = f"reused:{out_dir}"
    elif is_perch:
        fold_loaders = list(enumerate(dataloaders))
        perch_model = PerchEmbeddingModel(config.perch.preset, config.perch.batch_size, config.perch.load_workers)
        clip_frames = _perch_clip_frames(dataloaders)
        checkpoint_desc = f"perch_hoplite:{config.perch.preset}"
    else:
        fold_loaders = list(enumerate(dataloaders))
        checkpoint_dir = download_artifact(data_dir, run_id)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_params(flatten_params("embeddings", config.model_dump()))
        for fold, loaders in fold_loaders:
            out_path = out_dir / f"fold{fold}.h5"
            if config.reuse_embeddings:
                embeddings = _load_fold_embeddings(out_path)
            else:
                if is_perch:
                    embeddings = {split: perch_model.extract(clip_frames[fold][split]) for split in loaders}
                else:
                    best_path = checkpoint_dir / f"fold{fold}_best.pt"
                    checkpoint_path = best_path if best_path.exists() else checkpoint_dir / f"fold{fold}_last.pt"
                    checkpoint_desc = str(checkpoint_path)
                    source = HALLOEmbeddingModel(str(checkpoint_path))
                    embeddings = {split: source.extract(loader) for split, loader in loaders.items()}
                with h5py.File(out_path, "w") as h5:
                    for split, (split_embeddings, split_labels) in embeddings.items():
                        h5.create_dataset(f"{split}_embeddings", data=split_embeddings)
                        h5.create_dataset(f"{split}_labels", data=split_labels)

            with mlflow.start_run(nested=True, run_name=f"embeddings_fold{fold}"):
                mlflow.log_param("checkpoint_path", checkpoint_desc)
                mlflow.log_param("embeddings_path", str(out_path))
                mlflow.log_param("embedding_dim", next(iter(embeddings.values()))[0].shape[1])
                for split, (split_embeddings, _) in embeddings.items():
                    mlflow.log_param(f"{split}_n_samples", split_embeddings.shape[0])
                mlflow.log_artifact(str(out_path))

                if "train" in embeddings:
                    # run linear probing and store metric curves in mlflow post training linear layer
                    probe_embeddings = embeddings
                    class_names = (
                        list(config.reuse_classes) if loaders is None
                        else next(iter(loaders.values())).dataset.dataset.classes
                    )
                    if config.linear_probe_label_map:
                        probe_embeddings, class_names = remap_labels(
                            probe_embeddings, class_names, config.linear_probe_label_map,
                        )
                        mlflow.log_param("linear_probe_classes", class_names)
                    linear_probe = LinearProbe(epochs=config.linear_probe_epochs, lr=config.linear_probe_lr)
                    linear_probe_metrics, linear_probe_loss_curves = linear_probe.evaluate(probe_embeddings, class_names)
                    for curve_key, curve_values in linear_probe_loss_curves.items():
                        for epoch, value in enumerate(curve_values):
                            mlflow.log_metric(f"linear_probe_{curve_key}", value, step=epoch)
                    for key, value in linear_probe_metrics.items():
                        mlflow.log_metric(f"linear_probe_{key}", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Base config, e.g. configs/base.yaml.")
    parser.add_argument("--override", type=Path, default=None, help="Override config, deep-merged onto --config.")
    parser.add_argument(
        "--dataset-path", type=str, default=None,
        help="Dataset produced by preprocessor. Not needed when embeddings.reuse_embeddings is set.",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="model_trainer MLflow run that wrote the checkpoints to embed. Defaults to embeddings.mlflow_id in --config.",
    )
    args = parser.parse_args()
    pipeline_config = load_pipeline_config(args.config, args.override)
    run_id = args.run_id or pipeline_config.embeddings.mlflow_id
    if run_id is None:
        parser.error("--run-id is required unless embeddings.mlflow_id is set in --config.")

    configure_mlflow(pipeline_config)
    if pipeline_config.embeddings.reuse_embeddings:
        dataloaders = None
    elif args.dataset_path is None:
        parser.error("--dataset-path is required unless embeddings.reuse_embeddings is set.")
    else:
        with mlflow.start_run(run_id=run_id):
            dataloaders = build_dataloaders(args.dataset_path, pipeline_config.model_trainer.dataloader, pipeline_config.data_dir)
    generate_embeddings(pipeline_config.embeddings, dataloaders, run_id, pipeline_config.data_dir)


if __name__ == "__main__":
    main()
