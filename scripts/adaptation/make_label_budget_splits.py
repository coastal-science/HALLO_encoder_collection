"""Label-budget splits (experiment E): each site's pretraining-comparison 5-fold splits with only a fraction of
the labelled files kept, for the 10 / 25 / 50 % fine-tune tiers (100 % = the existing A / C5 / D runs).

Per fold, the train and val files (LocalPath groups) are subsampled separately, stratified by whether a file
contains an SRKW call, keeping ceil(frac * n) of each stratum; the test rows are untouched, so every tier is
scored on exactly the 100 % runs' test clips. Each fold's draw is one seeded shuffle, and a tier keeps a prefix of
it, so 10 % is a subset of 25 %, which is a subset of 50 %. Dropped rows are left blank, which load_saved_splits skips.

usage:
    python scripts/adaptation/make_label_budget_splits.py
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

DORI = Path("/home/noah/HALLO_encoder_collection/data_raw/DORI/20261022_190549")
SITES = ["dori_venus_strait_of_georgia", "limekiln", "tekteksen"]
FRACS = [10, 25, 50]
SEED = 20261027


def subsample(splits: pd.DataFrame, ann: pd.DataFrame, frac: int) -> pd.DataFrame:
    out = splits.copy()
    group = ann.set_index("uid").loc[splits.uid, "LocalPath"].to_numpy()
    srkw_files = set(ann.loc[ann.Labels == "SRKW", "LocalPath"])
    for k, fold in enumerate(c for c in splits.columns if c.startswith("fold_")):
        rng = np.random.default_rng(SEED + k)
        for split in ["train", "val"]:
            files = sorted(set(group[splits[fold] == split]))
            keep = set()
            for stratum in [[f for f in files if f in srkw_files], [f for f in files if f not in srkw_files]]:
                order = rng.permutation(len(stratum))  # drawn regardless of frac, so tiers nest
                keep |= {stratum[i] for i in order[: math.ceil(frac / 100 * len(stratum))]}
            out.loc[(splits[fold] == split).to_numpy() & ~np.isin(group, list(keep)), fold] = np.nan
    return out


def main() -> None:
    for site in SITES:
        splits = pd.read_csv(DORI / f"splits_pretrain_cmp_ft_{site}.csv")
        ann = pd.read_csv(DORI / f"annotations_bgcap10000_ft_{site}.csv", usecols=["uid", "LocalPath", "Labels"])
        for frac in FRACS:
            path = DORI / f"splits_label_budget_frac{frac}_ft_{site}.csv"
            subsample(splits, ann, frac).to_csv(path, index=False)
            print(path)


if __name__ == "__main__":
    main()
