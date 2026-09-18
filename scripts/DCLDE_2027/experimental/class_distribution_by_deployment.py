"""Class (Labels) distribution per deployment (Dataset) for the DCLDE 2027 annotations.

Writes three CSVs and prints the first two:
  --counts-out   deployment x class annotation counts, with a Total column and row
  --pct-out      the same, row-normalised to each deployment's class mix (%)
  --by-split-out long format Dataset,Split,Labels,count (needs --splits; Split is
                 'train'/'test', or 'unassigned' for uids missing from the split file)

Usage:
    python class_distribution_by_deployment.py [--drop-background]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE = Path("/home/noah/HALLO_encoder_collection/data_raw/DCLDE_2027/20260827_090049")
ANNOTATIONS = BASE / "annotations.csv"
SPLITS = BASE / "reconstructed_splits" / "splits_full_train.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    ap.add_argument("--splits", type=Path, default=SPLITS)
    ap.add_argument("--drop-background", action="store_true",
                    help="exclude the Background class from all three outputs")
    ap.add_argument("--counts-out", type=Path, default=Path("deployment_class_counts.csv"))
    ap.add_argument("--pct-out", type=Path, default=Path("deployment_class_pct.csv"))
    ap.add_argument("--by-split-out", type=Path, default=Path("deployment_class_by_split.csv"))
    args = ap.parse_args()

    a = pd.read_csv(args.annotations, low_memory=False, usecols=["uid", "Dataset", "Labels"])
    if args.drop_background:
        a = a[a["Labels"] != "Background"]

    # order deployments and classes by total volume (most first)
    dep_order = a["Dataset"].value_counts().index.tolist()
    cls_order = a["Labels"].value_counts().index.tolist()

    counts = (
        a.pivot_table(index="Dataset", columns="Labels", aggfunc="size", fill_value=0)
        .reindex(index=dep_order, columns=cls_order)
        .astype(int)
    )
    counts["Total"] = counts.sum(axis=1)
    counts.loc["ALL"] = counts.sum(axis=0)

    pct = counts.drop(columns="Total").div(counts["Total"], axis=0).mul(100).round(2)

    counts.to_csv(args.counts_out)
    pct.to_csv(args.pct_out)

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print("=== annotation counts: deployment x class ===")
        print(counts.to_string())
        print()
        print("=== row-normalised: each deployment's class mix (%) ===")
        print(pct.to_string())

    if args.splits and args.splits.exists():
        s = pd.read_csv(args.splits)                     # uid, fold_0
        m = a.merge(s, on="uid", how="left")
        m["fold_0"] = m["fold_0"].fillna("unassigned")
        by_split = (
            m.groupby(["Dataset", "fold_0", "Labels"], observed=True)
            .size().rename("count").reset_index()
            .rename(columns={"fold_0": "Split"})
            .sort_values(["Dataset", "Split", "count"], ascending=[True, True, False])
        )
        by_split.to_csv(args.by_split_out, index=False)
        print()
        print(f"wrote {args.counts_out}, {args.pct_out}, {args.by_split_out}")
    else:
        print()
        print(f"wrote {args.counts_out}, {args.pct_out}  (no --splits file: skipped by-split output)")


if __name__ == "__main__":
    main()
