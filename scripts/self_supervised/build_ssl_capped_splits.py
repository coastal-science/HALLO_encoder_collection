"""J's splits: I's SSL-labelled set (build_ssl_labelled_annotations.py) with SanctSound OC02 capped.

I's 3.65M rows are 84 % one site (SanctSound OC02, Olympic Coast: 1.44M humpback, 1.44M orca and 194k fin-whale
presence windows). This keeps every non-OC02 row and draws CAP OC02 rows, split as evenly as possible across OC02's
label groups (a group smaller than its share is kept whole and the rest goes to the other groups). Rows keep I's
train / val assignment and the draw is made per split, so the train / val ratio stays as in I.

Rows are written by uid in a splits.csv that load_saved_splits reads against I's HDF5; rows not listed are skipped,
so no new HDF5 is built.

usage:
    .venv/bin/python scripts/self_supervised/build_ssl_capped_splits.py [--cap 500000] [--seed 0]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/home/noah/HALLO_encoder_collection/data_raw/ssl_labelled/20261107_162232")
SITE = "OC02"


def even_shares(sizes: pd.Series, total: int) -> pd.Series:
    """Per-group counts summing to min(total, sizes.sum()), as even as possible, never above a group's size."""
    take, left = {}, min(total, int(sizes.sum()))
    ordered = sizes.sort_values()  # smallest first: each takes min(size, fair share of what is left)
    for i, (group, size) in enumerate(ordered.items()):
        take[group] = min(int(size), left // (len(ordered) - i))
        left -= take[group]
    return pd.Series(take)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ann = pd.read_csv(SRC / "annotations.csv", usecols=["uid", "Dataset", "Labels", "LocalPath"], low_memory=False)
    splits = pd.read_csv(SRC / "splits.csv")
    df = ann.merge(splits, on="uid", validate="one_to_one")
    is_site = (df.Dataset == "DORI-SanctSound") & df.LocalPath.str.contains(f"SanctSound_{SITE}_", regex=False)
    rng = np.random.default_rng(args.seed)

    kept = [df[~is_site]]
    site = df[is_site]
    for split, rows in site.groupby("fold_0"):
        cap = round(args.cap * len(rows) / len(site))
        take = even_shares(rows.Labels.value_counts(), cap)
        for label, n in take.items():
            group = rows[rows.Labels == label]
            kept.append(group.iloc[rng.choice(len(group), size=int(n), replace=False)])
    out = pd.concat(kept)

    path = SRC / f"splits_J_{SITE.lower()}cap{args.cap // 1000}k_seed{args.seed}.csv"
    out[["uid", "fold_0"]].sort_values("uid").to_csv(path, index=False)
    print(f"wrote {path}: {len(out):,} rows ({out.fold_0.value_counts().to_dict()})")
    print(f"{SITE} kept {int(is_site.sum()):,} -> {len(out) - int((~is_site).sum()):,}")
    print(out[out.uid.isin(site.uid)].Labels.value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
