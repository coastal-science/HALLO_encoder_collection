"""Caps the Background rows of the train split only, leaving val / test untouched
so they keep a realistic background rate.

The split has to be fixed before the cap (the cap needs to know which rows are
train), so this writes two files next to the source annotations csv:
  - annotations_bgcap<N>.csv: the source csv minus the capped-out train Background rows
  - splits_bgcap<N>.csv: uid -> train / val / test, for model_trainer.dataloader.splits_path

The split mirrors model_trainer.data_loader.compute_splits for the holdout-column
case: test_holdout_values of test_holdout_col go to test, val_size of the
remaining col_to_group_by groups go to val.

usage:
    python scripts/DORI/cap_train_background.py --annotations-csv data_raw/DORI/<run>/annotations.csv \
        --config configs/experiments/exp_0/00_base.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from encoder_pipeline.common.config_utils import load_pipeline_config

BACKGROUND_LABEL = "Background"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", type=Path, required=True, help="Source (uncapped) annotations csv.")
    parser.add_argument("--config", type=Path, required=True, help="Pipeline config to read the split params + classes_to_drop from.")
    parser.add_argument("--max-per-dataset", type=int, default=10_000, help="Max train Background rows kept per --cap-col value.")
    parser.add_argument("--cap-col", default="Dataset", help="Column the cap is applied within.")
    parser.add_argument("--seed", type=int, default=10, help="Seed for picking which train Background rows are kept.")
    args = parser.parse_args()

    config = load_pipeline_config(args.config, None)
    dataset_config, loader_config = config.preprocessor.dataset, config.model_trainer.dataloader
    if not (loader_config.test_holdout_col and loader_config.test_holdout_values and loader_config.col_to_group_by):
        raise ValueError("expects test_holdout_col, test_holdout_values and col_to_group_by to be set")
    if loader_config.n_folds > 1:
        raise ValueError("n_folds > 1 not supported")

    src_path = args.annotations_csv
    suffix = f"bgcap{args.max_per_dataset}"
    if suffix in src_path.name:
        raise ValueError(f"{src_path} is already capped -- pass the source annotations csv")
    df = pd.read_csv(src_path, low_memory=False)

    # rows the preprocessor would keep: not class-dropped, and with a file to group on
    usable = df[dataset_config.local_file_col].notna() & df[loader_config.col_to_group_by].notna()
    if dataset_config.classes_to_drop:
        usable &= ~df["Labels"].isin(dataset_config.classes_to_drop)

    is_test = usable & df[loader_config.test_holdout_col].isin(loader_config.test_holdout_values)
    groups = df[loader_config.col_to_group_by]
    _, val_groups = train_test_split(
        np.unique(groups[usable & ~is_test]), test_size=loader_config.val_size, random_state=loader_config.split_seed,
    )
    is_val = usable & ~is_test & groups.isin(val_groups)
    is_train = usable & ~is_test & ~is_val

    train_background = df[is_train & (df["Labels"] == BACKGROUND_LABEL)]
    kept_background = train_background.sample(frac=1, random_state=args.seed).groupby(args.cap_col).head(args.max_per_dataset)
    capped_out = train_background.index.difference(kept_background.index)

    split = pd.Series(pd.NA, index=df.index, dtype=object)
    split[is_train], split[is_val], split[is_test] = "train", "val", "test"
    keep = ~df.index.isin(capped_out)

    annotations_path = src_path.with_name(f"annotations_{suffix}.csv")
    splits_path = src_path.with_name(f"splits_{suffix}.csv")
    df[keep].to_csv(annotations_path, index=False)
    pd.DataFrame({"uid": df.loc[keep & usable, dataset_config.uid_col], "fold_0": split[keep & usable]}).to_csv(splits_path, index=False)

    summary = pd.DataFrame({
        "rows": split[keep].value_counts(),
        "background": split[keep & (df["Labels"] == BACKGROUND_LABEL)].value_counts(),
    })
    print(summary.to_string())
    print(f"train Background: {len(train_background):,} -> {len(kept_background):,}")
    print(f"annotations -> {annotations_path}")
    print(f"splits      -> {splits_path}")


if __name__ == "__main__":
    main()
