"""Builds the small DCLDE 2027 test dataset in this folder, for smoke-testing the embedding configs end to end.

Reads only the original DCLDE 2027 release under /data (readable by every user on the machine), not a processed
copy under data_raw/, and derives the processed columns (Labels, LocalPath, Duration, CenterTime, ... and the
"Background" windows) with scripts/DCLDE_2027/build_dclde_2027_annotations.py's own functions, so the rows have the
same schema and label logic as data_raw/DCLDE_2027/*/annotations.csv. The exception is the call-type columns
(CallType, CalltypeCategory): those come from CreateParentAnnotations.R's parent CSVs and are left empty here,
because the embedding pipeline doesn't use them.

Writes, next to this script:
  dclde_2027_test_annotations.csv  FILES_PER_CLASS audio files per class x ROWS_PER_FILE rows per file, for
                                   Background / HW / SRKW / TKW, every file used for only one class, files spread
                                   over the data providers. HW / SRKW / TKW rows are real annotations (uid = their
                                   row number in Annotations.csv). Background rows are 5 s windows at least 10 s
                                   from any annotated event in the file (the builder's "naive" method), with uid =
                                   BACKGROUND_UID_OFFSET + n, so uids are unique and stable
  dclde_2027_test_splits.csv       pinned 5-fold splits (uid, fold_0..fold_4), grouped by audio file: per class,
                                   TEST_FILES_PER_CLASS files are the fixed test set of every fold, the rest rotate
                                   through val, so each fold's train / val / test has every class and no file
                                   is in two splits.

The CSV only holds annotations: the audio is read from each row's LocalPath (under DCLDE_ROOT), so it must be
mounted where the pipeline runs.

Run from the repo root, with a venv that has this repo installed (encoder_pipeline, librosa, mlflow, tqdm):
    python embedding_models/test_dataset/build_test_dataset.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "DCLDE_2027"))  # the builder imports its sibling package by bare name
from build_dclde_2027_annotations import (  # noqa: E402
    BACKGROUND_LABEL, build_local_audio_index, derive_labels, select_naive_background_windows, to_annotations_schema,
)

DCLDE_ROOT = Path("/data/DCLDE_2027/dclde_2027_killer_whales")
SOURCE = DCLDE_ROOT / "Annotations.csv"
OUT_DIR = Path(__file__).resolve().parent
CLASSES = ["Background", "HW", "SRKW", "TKW"]
FILES_PER_CLASS = 20
ROWS_PER_FILE = 3
TEST_FILES_PER_CLASS = 2
N_FOLDS = 5
MIN_BEGIN_S = 3.0  # keep 5 s windows clear of 0 s at the start of each file
# the DCLDE builder's naive background-window settings (its argparse defaults)
BACKGROUND_WINDOW_S, BACKGROUND_HOP_S, BACKGROUND_EVENT_BUFFER_S = 5.0, 2.5, 10.0
BACKGROUND_UID_OFFSET = 10_000_000  # above every Annotations.csv row number
SEED = 0


def load_annotations() -> pd.DataFrame:
    """Annotations.csv with the processed columns the pipeline needs, derived as build_annotations_with_calltype does."""
    df = pd.read_csv(SOURCE, low_memory=False, index_col=0)
    df["uid"] = df.index.astype(np.int64)
    audio_index = build_local_audio_index(DCLDE_ROOT)
    df["LocalPath"] = df["Soundfile"].apply(lambda s: audio_index.get(Path(s).stem.lower()))
    df["LocalFileOk"] = df["LocalPath"].notna()
    df["CenterTime"] = (df["FileBeginSec"] + df["FileEndSec"]) / 2
    df["Duration"] = df["FileEndSec"] - df["FileBeginSec"]
    df["EcotypeCertain"] = df["Ecotype"].isin(["NRKW", "OKW", "SRKW", "TKW", "SAR"])
    df["Labels"] = derive_labels(df.copy())
    for col in ("CallType", "CalltypeCategory"):  # from CreateParentAnnotations.R's parent CSVs, not rebuilt here
        df[col] = np.nan
    df["HasQ"], df["CalltypeHasQ"] = False, False
    df["BackgroundMethod"] = np.nan
    return df


def provider_round_robin(files: pd.DataFrame) -> pd.DataFrame:
    """files (one row per LocalPath, with a provider column) shuffled, then interleaved across providers."""
    files = files.sample(frac=1, random_state=SEED)
    files["rank"] = files.groupby("provider").cumcount()
    return files.sort_values(["rank", "provider"])


def pick_event_rows(df: pd.DataFrame, label: str, used_files: set[str]) -> list[pd.DataFrame]:
    rows = df[(df["Labels"] == label) & ~df["LocalPath"].isin(used_files)]
    files = rows.groupby("LocalPath").agg(provider=("Provider", "first"), n=("uid", "size"))
    files = provider_round_robin(files[files["n"] >= ROWS_PER_FILE].reset_index())
    chosen = files.head(FILES_PER_CLASS)["LocalPath"]
    if len(chosen) < FILES_PER_CLASS:
        raise SystemExit(f"only {len(chosen)} usable files for {label}")
    used_files.update(chosen)
    return [rows[rows["LocalPath"] == path].sample(ROWS_PER_FILE, random_state=SEED) for path in chosen]


def pick_background_rows(df: pd.DataFrame, all_rows: pd.DataFrame, used_files: set[str]) -> list[pd.DataFrame]:
    """Naive background windows from files not used by another class, computed only for candidate files (in
    provider round-robin order) until FILES_PER_CLASS of them have ROWS_PER_FILE windows."""
    files = df[~df["LocalPath"].isin(used_files)].groupby("LocalPath").agg(provider=("Provider", "first")).reset_index()
    picked: list[pd.DataFrame] = []
    for path in provider_round_robin(files)["LocalPath"]:
        # every annotation in the file (not just the filtered ones), so windows keep clear of all its events
        windows = select_naive_background_windows(
            all_rows[all_rows["LocalPath"] == path], BACKGROUND_WINDOW_S, BACKGROUND_HOP_S, BACKGROUND_EVENT_BUFFER_S,
        )
        if len(windows) == 0 or (windows["FileBeginSec"] >= MIN_BEGIN_S).sum() < ROWS_PER_FILE:
            continue
        windows = windows[windows["FileBeginSec"] >= MIN_BEGIN_S].sample(ROWS_PER_FILE, random_state=SEED)
        picked.append(to_annotations_schema(windows, df.columns.tolist(), "naive"))
        used_files.add(path)
        if len(picked) == FILES_PER_CLASS:
            break
    if len(picked) < FILES_PER_CLASS:
        raise SystemExit(f"only {len(picked)} usable files for {BACKGROUND_LABEL}")
    background = pd.concat(picked, ignore_index=True)
    background["uid"] = BACKGROUND_UID_OFFSET + np.arange(len(background))
    return [background]


def main() -> None:
    rng = np.random.default_rng(SEED)
    all_rows = load_annotations()
    df = all_rows[all_rows["LocalFileOk"] & (all_rows["Duration"] > 0) & (all_rows["FileBeginSec"] >= MIN_BEGIN_S)]

    used_files: set[str] = set()
    picked = []
    # event classes rarest first, so the classes with the fewest files get first pick; background last
    for label in df.loc[df["Labels"].isin(CLASSES) & (df["Labels"] != BACKGROUND_LABEL), "Labels"].value_counts(ascending=True).index:
        picked += pick_event_rows(df, label, used_files)
    picked += pick_background_rows(df, all_rows, used_files)
    out = pd.concat(picked)[all_rows.columns].sort_values(["Labels", "LocalPath", "FileBeginSec"]).reset_index(drop=True)
    assert out["uid"].is_unique

    splits = pd.DataFrame({"uid": out["uid"]})
    for fold in range(N_FOLDS):
        splits[f"fold_{fold}"] = "train"
    for label in CLASSES:
        files = out.loc[out["Labels"] == label, "LocalPath"].unique()
        files = files[rng.permutation(len(files))]
        test_files, rotating = files[:TEST_FILES_PER_CLASS], files[TEST_FILES_PER_CLASS:]
        for fold in range(N_FOLDS):
            val_files = rotating[fold::N_FOLDS]
            splits.loc[out["LocalPath"].isin(test_files), f"fold_{fold}"] = "test"
            splits.loc[out["LocalPath"].isin(val_files), f"fold_{fold}"] = "val"

    out.to_csv(OUT_DIR / "dclde_2027_test_annotations.csv", index=False)
    splits.to_csv(OUT_DIR / "dclde_2027_test_splits.csv", index=False)
    print(out["Labels"].value_counts().to_dict(), "| files:", out["LocalPath"].nunique(),
          "| providers:", out["Provider"].value_counts().to_dict())
    print(splits.drop(columns="uid").apply(pd.Series.value_counts).to_dict())


if __name__ == "__main__":
    main()
