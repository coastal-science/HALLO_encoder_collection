"""Builds a simple file-level 70/15/15 train/val/test split from a DCLDE
annotations csv, using experimental/MAI_preprocessing.py's filtering + split
logic. Writes the filtered annotations csv and a splits_70_15_15.csv into a
timestamped run dir under data_raw/DCLDE_2027, logged to MLflow's shared
'Datasets' experiment.

Background rows are kept in the annotations csv but excluded from the split
(their "split" value is left blank) -- background/NonBio window selection is
its own process, done elsewhere (see build_dclde_2027_annotations.py's
background_method), not here.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_NAME = "DCLDE_2027"
MLFLOW_EXPERIMENT_NAME = f"Datasets/{DATASET_NAME}"
BACKGROUND_LABEL = "Background"
DEFAULT_ANNOTATIONS_PATH = Path("/home/noah/HALLO_encoder_collection/data_raw/DCLDE_2027/20260827_090049/annotations.csv")


def filter_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """Drops exact dupes and the KW_certain==0 / zero-duration / file-level rows MAI_preprocessing.py flags as noise."""
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[~(df["KW_certain"] == 0.0)]
    df = df[df["Duration"] != 0.0]
    df = df[df["AnnotationLevel"] != "File"]
    return df.reset_index(drop=True)


def split_by_file(df: pd.DataFrame, val_size: float, test_size: float, random_state: int) -> pd.Series:
    """File-level split stratified by each file's dominant Labels value, so
    class proportions hold in each split. Background rows are excluded from
    splitting and left NA."""
    eligible = df["Labels"] != BACKGROUND_LABEL
    file_label = df[eligible].groupby("Soundfile")["Labels"].agg(lambda s: s.value_counts().idxmax())

    train_files, rest_files = train_test_split(
        file_label.index, test_size=val_size + test_size, stratify=file_label, random_state=random_state,
    )
    val_files, test_files = train_test_split(
        rest_files, test_size=test_size / (val_size + test_size),
        stratify=file_label.loc[rest_files], random_state=random_state,
    )
    file_to_split = {**dict.fromkeys(train_files, "train"), **dict.fromkeys(val_files, "val"), **dict.fromkeys(test_files, "test")}

    split = pd.Series(pd.NA, index=df.index, dtype="object")
    split[eligible] = df.loc[eligible, "Soundfile"].map(file_to_split)
    return split


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-path", type=Path, default=DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data_raw" / DATASET_NAME)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.data_dir / f"70_15_15_split_{timestamp}"
    run_dir.mkdir(parents=True)

    if args.mlflow_tracking_uri is not None:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_dir.name):
        mlflow.log_params({
            "annotations_path": str(args.annotations_path),
            "run_dir": str(run_dir),
            "val_size": args.val_size,
            "test_size": args.test_size,
            "random_state": args.random_state,
        })

        raw = pd.read_csv(args.annotations_path, low_memory=False)
        filtered = filter_annotations(raw)
        filtered["split"] = split_by_file(filtered, args.val_size, args.test_size, args.random_state)

        annotations_out = run_dir / "annotations.csv"
        filtered.to_csv(annotations_out, index=False)
        mlflow.log_param("annotations_out", str(annotations_out))
        mlflow.log_metric("n_rows_raw", len(raw))
        mlflow.log_metric("n_rows_filtered", len(filtered))
        mlflow.log_metric("n_background_rows", int((filtered["Labels"] == BACKGROUND_LABEL).sum()))

        assigned = filtered["split"].notna()
        splits_out = run_dir / "splits_70_15_15.csv"
        splits_df = filtered.loc[assigned, ["uid", "split"]].rename(columns={"split": "fold_0"})
        splits_df.to_csv(splits_out, index=False)
        mlflow.log_param("splits_out", str(splits_out))
        split_counts = splits_df["fold_0"].value_counts().to_dict()
        for name, count in split_counts.items():
            mlflow.log_metric(f"n_{name}", count)

        print(f"Saved {len(filtered):,} rows -> {annotations_out}")
        print(f"splits_70_15_15.csv: {split_counts} ({len(filtered) - len(splits_df)} rows dropped) -> {splits_out}")


if __name__ == "__main__":
    main()
