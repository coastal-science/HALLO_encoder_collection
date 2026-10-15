"""DCLDE 2027 + DORI annotations / splits for SSL pretraining.

The DORI annotations csv is built on top of the DCLDE one and already holds
every DCLDE row (uids are unique across both), so nothing is concatenated: the
SSL set is the train / val rows of exp_0's split of that one csv. exp_1's
fine-tuning splits only re-split exp_0's test rows, so dropping exp_0's test
rows keeps every exp_0 / exp_1 test clip out of pretraining:
  - train: exp_0's train rows (train Background already capped to 10k per Dataset)
  - val:   exp_0's val rows, with Background randomly subsampled to train's
           Background share (exp_0's val is uncapped, 85 % Background)
Also dropped: rows in an audio file that holds any exp_0 / exp_1 test row, and
repeats of the same clip (same file and window under two labels).
Source is the corpus the audio file sits in.

usage:
    python scripts/self_supervised/build_ssl_dclde_dori_annotations.py
"""
import glob
import time
from pathlib import Path

import pandas as pd

ANNOTATIONS = "data_raw/DORI/20261005_183013/annotations_bgcap10000.csv"
SPLITS = "data_raw/DORI/20261005_183013/splits_bgcap10000.csv"
FINETUNE_SPLITS = "data_raw/DORI/20261005_183013/splits_bgcap10000_ft_*.csv"
OUT_ROOT = "data_raw/ssl_dclde_dori"
SEED = 10

if __name__ == "__main__":
    rows = pd.read_csv(ANNOTATIONS, low_memory=False).merge(pd.read_csv(SPLITS), on="uid")
    finetune = pd.concat(pd.read_csv(path) for path in glob.glob(FINETUNE_SPLITS))
    is_test = (rows["fold_0"] == "test") | rows["uid"].isin(finetune.loc[finetune["fold_0"] == "test", "uid"])
    # exp_1 trains on rows that are test in exp_0, so those count as test here too
    is_test |= rows["uid"].isin(finetune.loc[finetune["fold_0"] == "train", "uid"])

    ssl = rows[~is_test & rows["fold_0"].isin(["train", "val"])]
    ssl = ssl[~ssl["LocalPath"].isin(rows.loc[is_test, "LocalPath"])]
    ssl = ssl.drop_duplicates(["LocalPath", "FileBeginSec", "FileEndSec"])
    ssl = ssl.assign(Source=ssl["LocalPath"].str.startswith("/data/DORI").map({True: "dori", False: "dclde"}))

    train, val = ssl[ssl["fold_0"] == "train"], ssl[ssl["fold_0"] == "val"]
    val_calls, val_background = val[val["Labels"] != "Background"], val[val["Labels"] == "Background"]
    background_share = (train["Labels"] == "Background").mean()
    n_background = round(len(val_calls) * background_share / (1 - background_share))
    ssl = pd.concat([train, val_calls, val_background.sample(n=n_background, random_state=SEED)]).sort_values("uid")

    assert ssl["uid"].is_unique
    assert not ssl["uid"].isin(rows.loc[is_test, "uid"]).any()
    assert not ssl["LocalPath"].isin(rows.loc[is_test, "LocalPath"]).any()

    out_dir = Path(OUT_ROOT) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    ssl.drop(columns="fold_0").to_csv(out_dir / "annotations.csv", index=False)
    ssl[["uid", "fold_0"]].to_csv(out_dir / "splits.csv", index=False)
    print(pd.crosstab(ssl["Source"], ssl["fold_0"], margins=True).to_string())
    print(pd.crosstab([ssl["Source"], ssl["Labels"]], ssl["fold_0"], margins=True).to_string())
    print(f"files: {ssl['LocalPath'].nunique():,} -> {out_dir}")
