import argparse
import json
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
from encoder_pipeline.evaluation.predictions import predictions_frame
from encoder_pipeline.model_trainer.data_loader import (
    SpectrogramDataset, build_dataloaders, load_uids, ordered_loader,
)


def _perch_clip_frames(dataloaders: list[dict[str, DataLoader]]) -> list[dict[str, pd.DataFrame]]:
    """Per fold/split, the uid/LocalPath/FileBeginSec/Duration/label rows PerchEmbeddingModel
    needs, pulled from the preprocessor HDF5 by one epoch of each loader's sampler,
    so the train split gets the same oversampling HALLOEmbeddingModel sees."""
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
    uids = load_uids(base.hdf5_path)
    return [
        {
            split: pd.DataFrame({
                "uid": uids[idx], "LocalPath": local_path[idx], "FileBeginSec": begin[idx],
                "Duration": duration[idx], "label": labels[idx],
            })
            for split, loader in loaders.items()
            for idx in [np.asarray(loader.dataset.indices)[np.fromiter(loader.sampler, dtype=np.int64)]]
        }
        for loaders in dataloaders
    ]


def _load_fold_embeddings(
    path: Path, class_label_map: dict[str, str] | None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str], dict[str, np.ndarray]]:
    """The {split: (embeddings, labels)} a prior run wrote to fold<n>.h5, its class
    list, and the {split: uids} of those rows (empty for files written before uids
    were stored). """
    with h5py.File(path, "r") as h5:
        if "class_label_map" in h5.attrs and json.loads(h5.attrs["class_label_map"]) != (class_label_map or {}):
            raise ValueError(
                f"{path} was written with class_label_map={h5.attrs['class_label_map']}, config has {class_label_map}"
            )
        splits = sorted(k[: -len("_embeddings")] for k in h5 if k.endswith("_embeddings"))
        embeddings = {s: (h5[f"{s}_embeddings"][:], h5[f"{s}_labels"][:]) for s in splits}
        uids = {
            s: dset.asstr()[:] if h5py.check_string_dtype(dset.dtype) else dset[:]
            for s in splits if f"{s}_uids" in h5
            for dset in [h5[f"{s}_uids"]]
        }
        classes = list(h5.attrs["classes"]) if "classes" in h5.attrs else None
    if classes is None:
        dataset_path = mlflow.get_run(path.parent.name).data.params["dataset_path"]
        classes = SpectrogramDataset(dataset_path, class_label_map=class_label_map).classes
    return embeddings, classes, uids


def _log_linear_probe(
    config: EmbeddingsConfig, embeddings: dict[str, tuple[np.ndarray, np.ndarray]], class_names: list[str],
    uids: dict[str, np.ndarray], predictions_path: Path,
) -> None:
    """Trains the linear probe on the train split and logs its curves, metrics and
    per-sample test/val predictions (written to predictions_path) to the active run."""
    if config.linear_probe_label_map:
        embeddings, class_names = remap_labels(embeddings, class_names, config.linear_probe_label_map)
        mlflow.log_param("linear_probe_classes", class_names)
    linear_probe = LinearProbe(epochs=config.linear_probe_epochs, lr=config.linear_probe_lr)
    linear_probe_metrics, linear_probe_loss_curves, scores = linear_probe.evaluate(embeddings, class_names)
    for curve_key, curve_values in linear_probe_loss_curves.items():
        for epoch, value in enumerate(curve_values):
            mlflow.log_metric(f"linear_probe_{curve_key}", value, step=epoch)
    for key, value in linear_probe_metrics.items():
        mlflow.log_metric(f"linear_probe_{key}", value)
    frames = [
        predictions_frame(split, uids.get(split), embeddings[split][1], split_scores, class_names)
        for split, split_scores in scores.items() if split != "train"
    ]
    if frames:
        pd.concat(frames).to_csv(predictions_path, index=False)
        mlflow.log_artifact(str(predictions_path))


def rerun_linear_probe(config: EmbeddingsConfig, class_label_map: dict[str, str] | None = None) -> None:
    """Linear probe on config.reuse_embeddings_path, logged as a new nested run under
    the run that wrote it (the file's parent directory name)."""
    path = Path(config.reuse_embeddings_path)
    embeddings, class_names, uids = _load_fold_embeddings(path, class_label_map)
    with mlflow.start_run(run_id=path.parent.name):
        with mlflow.start_run(nested=True, run_name=f"linear_probe_{path.stem}"):
            mlflow.log_params(flatten_params("embeddings", config.model_dump()))
            mlflow.log_param("classes", class_names)
            mlflow.log_param("class_label_map", class_label_map)
            _log_linear_probe(
                config, embeddings, class_names, uids,
                path.with_name(f"{path.stem}_linear_probe_predictions.csv"),
            )


def generate_embeddings(
    config: EmbeddingsConfig, dataloaders: list[dict[str, DataLoader]], run_id: str, data_dir: str,
    class_label_map: dict[str, str] | None = None,
) -> None:
    """"""
    out_dir = Path(f"{data_dir}/embeddings/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    is_perch = config.source == "perch_hoplite"
    if is_perch:
        perch_model = PerchEmbeddingModel(
            config.perch.preset, config.perch.batch_size, config.perch.load_workers, config.perch.chunk_size
        )
        clip_frames = _perch_clip_frames(dataloaders)
        checkpoint_desc = f"perch_hoplite:{config.perch.preset}"
    else:
        checkpoint_dir = download_artifact(data_dir, config.checkpoint_mlflow_id or run_id)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_params(flatten_params("embeddings", config.model_dump()))
        for fold, loaders in enumerate(dataloaders):
            out_path = out_dir / f"fold{fold}.h5"
            class_names = next(iter(loaders.values())).dataset.dataset.classes
            embeddings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            uids: dict[str, np.ndarray] = {}
            if is_perch:
                for split in loaders:
                    clips = clip_frames[fold][split]
                    split_embeddings, split_labels, kept = perch_model.extract(clips)
                    embeddings[split] = (split_embeddings, split_labels)
                    uids[split] = clips["uid"].to_numpy()[kept]
            else:
                best_path = checkpoint_dir / f"fold{fold}_best.pt"
                checkpoint_path = best_path if best_path.exists() else checkpoint_dir / f"fold{fold}_last.pt"
                checkpoint_desc = str(checkpoint_path)
                source = HALLOEmbeddingModel(str(checkpoint_path))
                all_uids = load_uids(next(iter(loaders.values())).dataset.dataset.hdf5_path)
                for split, loader in loaders.items():
                    # ordered, so every embedding row can be traced back to its uid
                    ordered, row_idx = ordered_loader(loader)
                    embeddings[split] = source.extract(ordered)
                    uids[split] = all_uids[row_idx]
            with h5py.File(out_path, "w") as h5:
                h5.attrs["classes"] = class_names
                h5.attrs["class_label_map"] = json.dumps(class_label_map or {})
                for split, (split_embeddings, split_labels) in embeddings.items():
                    h5.create_dataset(f"{split}_embeddings", data=split_embeddings)
                    h5.create_dataset(f"{split}_labels", data=split_labels)
                    is_str = uids[split].dtype.kind in "OUS"
                    h5.create_dataset(
                        f"{split}_uids", data=uids[split], dtype=h5py.string_dtype() if is_str else None,
                    )

            with mlflow.start_run(nested=True, run_name=f"embeddings_fold{fold}"):
                mlflow.log_param("checkpoint_path", checkpoint_desc)
                mlflow.log_param("embeddings_path", str(out_path))
                mlflow.log_param("classes", class_names)
                mlflow.log_param("class_label_map", class_label_map)
                mlflow.log_param("embedding_dim", next(iter(embeddings.values()))[0].shape[1])
                for split, (split_embeddings, _) in embeddings.items():
                    mlflow.log_param(f"{split}_n_samples", split_embeddings.shape[0])
                mlflow.log_artifact(str(out_path))

                if "train" in embeddings:
                    # run linear probing and store metric curves in mlflow post training linear layer
                    _log_linear_probe(
                        config, embeddings, class_names, uids,
                        out_dir / f"fold{fold}_linear_probe_predictions.csv",
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Base config, e.g. configs/base.yaml.")
    parser.add_argument("--override", type=Path, default=None, help="Override config, deep-merged onto --config.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset produced by preprocessor.")
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
    with mlflow.start_run(run_id=run_id):
        dataloaders = build_dataloaders(args.dataset_path, pipeline_config.model_trainer.dataloader, pipeline_config.data_dir)
    generate_embeddings(
        pipeline_config.embeddings, dataloaders, run_id, pipeline_config.data_dir,
        pipeline_config.model_trainer.dataloader.class_label_map,
    )

if __name__ == "__main__":
    main()
