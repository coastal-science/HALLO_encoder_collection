"""Writes the exact train / val / test splits a pipeline config trains on, as a uid -> fold_N splits.csv.

Runs the same steps as model_trainer.data_loader.build_dataloaders -- load or compute the splits on the
config's preprocessor hdf5 (which must already exist), holdout_val_from_train, then the
drop_unplaceable_windows / train_overlap_tol_s filters -- without training or MLflow. Point another
config's dataloader.splits_path at the output to evaluate on identical folds even when its own hdf5
differs, e.g. a perch_hoplite metadata-only hdf5 built after the NaN-window guard, which lacks rows (and
whole files) the spectrogram hdf5 kept and so would compute different folds.

usage:
    python scripts/DORI/export_config_splits.py --config <base.yaml> --override <step.yaml> --out <splits.csv>
"""
import argparse
from pathlib import Path

import pandas as pd

from encoder_pipeline.common.config_utils import load_pipeline_config
from encoder_pipeline.model_trainer.data_loader import (
    compute_splits, drop_train_overlap, drop_unplaceable_windows, holdout_val_from_train, load_saved_splits, load_uids,
)
from encoder_pipeline.preprocessor.dataset import Dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = load_pipeline_config(args.config, args.override)
    pre, loader_config = config.preprocessor, config.model_trainer.dataloader
    dataset = Dataset(pre.spectrogram, pre.audio_file, pre.annotation, pre.dataset, config.data_dir, pre.run_name)
    if not dataset.is_materialized:
        raise FileNotFoundError(f"{dataset.out_file} doesn't exist -- run the config's preprocessor first")
    hdf5_path = dataset.out_file

    splits = (
        load_saved_splits(hdf5_path, loader_config.splits_path) if loader_config.splits_path
        else compute_splits(hdf5_path, loader_config)
    )
    splits = holdout_val_from_train(hdf5_path, splits, loader_config)
    if loader_config.drop_unplaceable_windows:
        splits = drop_unplaceable_windows(hdf5_path, splits)
    if loader_config.train_overlap_tol_s is not None:
        splits = drop_train_overlap(hdf5_path, splits, loader_config)

    # only the rows some split uses, unlike save_splits, so a reader's hdf5 can't pick up rows these folds left out
    uids = load_uids(hdf5_path)
    folds = []
    for fold, split in enumerate(splits):
        folds.append(pd.concat(
            pd.DataFrame({"uid": uids[idx], "fold": f"fold_{fold}", "split": name}) for name, idx in split.items()
        ))
    out = pd.concat(folds).pivot(index="uid", columns="fold", values="split").reset_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"{hdf5_path} -> {args.out}")
    for col in [c for c in out.columns if c.startswith("fold_")]:
        print(f"  {col}: {out[col].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
