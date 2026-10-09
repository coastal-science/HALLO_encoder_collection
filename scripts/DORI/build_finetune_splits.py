"""Per-deployment fine-tuning splits on top of an existing held-out-deployment split.

For each test_holdout_values deployment (minus --exclude), writes a splits csv
next to --splits-csv where:
  - train: 1 - test_size of that deployment's rows (all of them test in --splits-csv)
  - test:  the remaining test_size of that deployment
  - val:   --splits-csv's val rows, unchanged
Every other row (the source train split, the other held-out deployments) is left
out, so model_trainer.dataloader.splits_path skips it.

The train / test cut is grouped by col_to_group_by (no file on both sides) and
stratified by the class_label_map'd label.

usage:
    python scripts/DORI/build_finetune_splits.py \
        --annotations-csv data_raw/DORI/<run>/annotations_bgcap10000.csv \
        --splits-csv data_raw/DORI/<run>/splits_bgcap10000.csv \
        --config configs/experiments/exp_1/00_base.yaml
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from encoder_pipeline.common.config_utils import load_pipeline_config


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", type=Path, required=True, help="Annotations csv --splits-csv was built for.")
    parser.add_argument("--splits-csv", type=Path, required=True, help="Source uid -> train / val / test split.")
    parser.add_argument("--config", type=Path, required=True, help="Pipeline config to read the holdout / grouping params from.")
    parser.add_argument("--exclude", nargs="*", default=["DORI VENUS (Saanich)"], help="test_holdout_values to skip.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Fraction of each deployment kept as test.")
    args = parser.parse_args()

    config = load_pipeline_config(args.config, None)
    dataset_config, loader_config = config.preprocessor.dataset, config.model_trainer.dataloader
    if not (loader_config.test_holdout_col and loader_config.test_holdout_values and loader_config.col_to_group_by):
        raise ValueError("expects test_holdout_col, test_holdout_values and col_to_group_by to be set")

    df = pd.read_csv(
        args.annotations_csv, low_memory=False,
        usecols=[dataset_config.uid_col, "Labels", loader_config.test_holdout_col, loader_config.col_to_group_by],
    ).merge(pd.read_csv(args.splits_csv), left_on=dataset_config.uid_col, right_on="uid")
    df["label"] = df["Labels"].replace(loader_config.class_label_map or {})
    val = df[df["fold_0"] == "val"]

    n_splits = round(1 / args.test_size)
    for deployment in loader_config.test_holdout_values:
        if deployment in args.exclude:
            continue
        rows = df[(df[loader_config.test_holdout_col] == deployment) & (df["fold_0"] == "test")]
        kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=loader_config.split_seed)
        train_pos, test_pos = next(kfold.split(rows, rows["label"], rows[loader_config.col_to_group_by]))
        train, test = rows.iloc[train_pos], rows.iloc[test_pos]

        out_path = args.splits_csv.with_name(f"{args.splits_csv.stem}_ft_{slugify(deployment)}.csv")
        pd.concat([
            pd.DataFrame({"uid": train["uid"], "fold_0": "train"}),
            pd.DataFrame({"uid": val["uid"], "fold_0": "val"}),
            pd.DataFrame({"uid": test["uid"], "fold_0": "test"}),
        ]).to_csv(out_path, index=False)

        summary = pd.DataFrame({
            "train": train["label"].value_counts(), "test": test["label"].value_counts(),
        }).fillna(0).astype(int)
        summary.loc["rows"] = [len(train), len(test)]
        summary.loc["files"] = [
            train[loader_config.col_to_group_by].nunique(), test[loader_config.col_to_group_by].nunique(),
        ]
        print(f"\n{deployment} (val: {len(val):,} rows, unchanged) -> {out_path}")
        print(summary.to_string())


if __name__ == "__main__":
    main()
