"""Per-deployment fine-tuning annotations from an existing held-out-deployment split.

For each test_holdout_values deployment (minus --exclude), writes an annotations
csv next to --annotations-csv holding only that deployment's rows (all of them
test in --splits-csv), every column unchanged. Point preprocessor.dataset.annotations_csv
at it and let model_trainer.dataloader.n_folds do the k-fold split.

usage:
    python scripts/DORI/build_finetune_annotations.py \
        --annotations-csv data_raw/DORI/<run>/annotations_bgcap10000.csv \
        --splits-csv data_raw/DORI/<run>/splits_bgcap10000.csv \
        --config configs/self_supervised_experiments/exp_0/00_base.yaml
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from encoder_pipeline.common.config_utils import load_pipeline_config


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", type=Path, required=True, help="Annotations csv --splits-csv was built for.")
    parser.add_argument("--splits-csv", type=Path, required=True, help="Source uid -> train / val / test split.")
    parser.add_argument("--config", type=Path, required=True, help="Pipeline config to read the holdout params from.")
    parser.add_argument("--exclude", nargs="*", default=["DORI VENUS (Saanich)"], help="test_holdout_values to skip.")
    args = parser.parse_args()

    config = load_pipeline_config(args.config, None)
    dataset_config, loader_config = config.preprocessor.dataset, config.model_trainer.dataloader
    if not (loader_config.test_holdout_col and loader_config.test_holdout_values):
        raise ValueError("expects test_holdout_col and test_holdout_values to be set")

    df = pd.read_csv(args.annotations_csv, low_memory=False)
    splits = pd.read_csv(args.splits_csv)
    test_uids = splits.loc[splits["fold_0"] == "test", "uid"]
    df = df[df[dataset_config.uid_col].isin(test_uids)]

    for deployment in loader_config.test_holdout_values:
        if deployment in args.exclude:
            continue
        rows = df[df[loader_config.test_holdout_col] == deployment]
        out_path = args.annotations_csv.with_name(f"{args.annotations_csv.stem}_ft_{slugify(deployment)}.csv")
        rows.to_csv(out_path, index=False)
        print(f"\n{deployment}: {len(rows):,} rows, {rows[dataset_config.local_file_col].nunique():,} files -> {out_path}")
        print(rows["Labels"].value_counts().to_string())


if __name__ == "__main__":
    main()
