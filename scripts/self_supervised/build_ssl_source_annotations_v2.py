"""SSL annotations / splits over the labelled DORI v2 source clips only (configs/pretraining_comparison's B).

The control for exp_1's SSL set (every audio file tiled, ~4.6M windows): the same masked-patch objective, but
on exactly the clips the supervised source model (A) trains and validates on, labels ignored. A vs B
isolates the objective at equal data; B vs exp_1's SSL isolates the extra unlabelled audio.
Same recipe as build_ssl_dclde_dori_annotations.py (exp_0's v1 SSL set), on DORI v2:
  - train: the v2 split's train rows (train Background already capped to 10k per Dataset)
  - val:   the v2 split's val rows, with Background randomly subsampled to train's Background share
Left out: the held-out test deployments, rows in a file that holds a test row, repeats of the same clip (same
file and window under two labels), and the rows with a NaN FileBeginSec / Duration, which the supervised
source model drops too (model_trainer.dataloader.drop_unplaceable_windows).

usage:
    python scripts/self_supervised/build_ssl_source_annotations_v2.py
"""
import time
from pathlib import Path

import pandas as pd

ANNOTATIONS = "data_raw/DORI/20261022_190549/annotations_bgcap10000.csv"
SPLITS = "data_raw/DORI/20261022_190549/splits_bgcap10000.csv"
OUT_ROOT = "data_raw/ssl_dori_source"
SEED = 10

if __name__ == "__main__":
    rows = pd.read_csv(ANNOTATIONS, low_memory=False).merge(pd.read_csv(SPLITS), on="uid")
    is_test = rows["fold_0"] == "test"

    unplaceable = pd.to_numeric(rows["FileBeginSec"], errors="coerce").isna() | pd.to_numeric(rows["Duration"], errors="coerce").isna()

    ssl = rows[rows["fold_0"].isin(["train", "val"]) & ~unplaceable]
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
    assert pd.to_numeric(ssl["Duration"], errors="coerce").notna().all()

    out_dir = Path(OUT_ROOT) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    ssl.drop(columns="fold_0").to_csv(out_dir / "annotations.csv", index=False)
    ssl[["uid", "fold_0"]].to_csv(out_dir / "splits.csv", index=False)
    print(pd.crosstab(ssl["Source"], ssl["fold_0"], margins=True).to_string())
    print(pd.crosstab([ssl["Source"], ssl["Labels"]], ssl["fold_0"], margins=True).to_string())
    print(f"train rows dropped vs the supervised train split: {int((rows['fold_0'] == 'train').sum()) - len(train):,}")
    print(f"files: {ssl['LocalPath'].nunique():,} -> {out_dir}")
