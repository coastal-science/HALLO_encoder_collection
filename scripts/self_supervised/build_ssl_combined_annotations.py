"""Combined DCLDE 2027 + blue_fin_antartica annotations / splits for SSL pretraining.

Only rows the downstream experiments train or validate on are used, so no
downstream test clip is ever seen in pretraining:
  - DCLDE: every train / val row of splits_70_15_15.csv (supervised_2 / supervised_3)
  - blue_fin_antartica: adaptation exp1's train / val rows (also exp4's), with
    Background randomly subsampled so each split matches DCLDE's row count
uids are prefixed with their source, as both datasets number rows from 0.

usage:
    python scripts/self_supervised/build_ssl_combined_annotations.py
"""
import time
from pathlib import Path

import pandas as pd

DCLDE_ANNOTATIONS = "data_raw/DCLDE_2027/70_15_15_split_20260921_075227/annotations.csv"
DCLDE_SPLITS = "data_raw/DCLDE_2027/70_15_15_split_20260921_075227/splits_70_15_15.csv"
BLUE_FIN_ANNOTATIONS = "data_raw/blue_fin_antartica/20261007_104040/annotations.csv"
BLUE_FIN_SPLITS = "data/model_trainer/9d378a789945439389a7f5a92348d035/splits.csv"
OUT_ROOT = "data_raw/ssl_dclde_blue_fin"
COLUMNS = ["uid", "Source", "Dataset", "Labels", "LocalPath", "FileBeginSec", "FileEndSec", "Duration", "CenterTime"]
SEED = 10

if __name__ == "__main__":
    dclde = pd.read_csv(DCLDE_ANNOTATIONS, low_memory=False).merge(pd.read_csv(DCLDE_SPLITS), on="uid")
    dclde = dclde[dclde["fold_0"].isin(["train", "val"])].assign(Source="dclde")

    blue_fin = pd.read_csv(BLUE_FIN_ANNOTATIONS, low_memory=False).merge(pd.read_csv(BLUE_FIN_SPLITS), on="uid")
    blue_fin = blue_fin[blue_fin["fold_0"].isin(["train", "val"])].assign(Source="blue_fin")
    kept = []
    for split, rows in blue_fin.groupby("fold_0"):
        calls, background = rows[rows["Labels"] != "Background"], rows[rows["Labels"] == "Background"]
        n_background = (dclde["fold_0"] == split).sum() - len(calls)
        kept += [calls, background.sample(n=n_background, random_state=SEED)]
    blue_fin = pd.concat(kept)

    combined = pd.concat([dclde, blue_fin], ignore_index=True)
    combined["uid"] = combined["Source"] + "_" + combined["uid"].astype(str)

    out_dir = Path(OUT_ROOT) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    combined[COLUMNS].to_csv(out_dir / "annotations.csv", index=False)
    combined[["uid", "fold_0"]].to_csv(out_dir / "splits.csv", index=False)
    print(pd.crosstab([combined["Source"], combined["Labels"]], combined["fold_0"], margins=True).to_string())
    print(f"files: {combined['LocalPath'].nunique():,} -> {out_dir}")
