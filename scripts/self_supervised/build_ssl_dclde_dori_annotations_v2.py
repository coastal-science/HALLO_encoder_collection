"""DCLDE 2027 + DORI (+ ONC background) annotations / splits for SSL pretraining, v2.

v1 only used the annotated windows of exp_0's train / val rows. SSL needs no labels, so v2 tiles every
audio file on disk into non-overlapping windows instead -- annotated or not, including the held-out
deployments (no labels are seen in pretraining, so this does not leak test labels):
  - DCLDE 2027 killer whale audio (every provider / site folder)
  - DORI-ONC, DORI-OOI and DORI-Orcasound audio (ONC-Benchmark is left out: it is a benchmark test set)
  - the ONC Strait of Georgia background files in scripts/DORI/download_onc_background.py's manifest
Windows are WINDOW_SEC long (2 * the preprocessor's annotation.time_offset); a file's last partial window
is dropped. val is VAL_FRACTION of the files (grouped by file, so no file is on both sides).

Every row is labelled 'Unlabelled'. ~3.7k h of audio gives ~4.5M rows: set preprocessor.dataset.spec_dtype
to float16 (~1.3 TB of HDF5 instead of ~2.6 TB).

usage:
    python scripts/self_supervised/build_ssl_dclde_dori_annotations_v2.py
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

AUDIO_DIRS = {
    "dclde": "/data/DCLDE_2027/dclde_2027_killer_whales",
    "dori": ["/data/DORI/DORI-ONC", "/data/DORI/DORI-OOI", "/data/DORI/DORI-Orcasound"],
}
ONC_BACKGROUND_MANIFEST = "/data/DORI/ONC-SoG-background/manifest.csv"
AUDIO_EXTENSIONS = (".wav", ".flac")
WINDOW_SEC = 3.0  # = 2 * preprocessor.annotation.time_offset in configs/self_supervised_experiments/exp_0
VAL_FRACTION = 0.05
LABEL = "Unlabelled"
OUT_ROOT = "data_raw/ssl_dclde_dori"
SEED = 10


def audio_files(root: str) -> list[str]:
    return sorted(
        os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.lower().endswith(AUDIO_EXTENSIONS)
    )


def duration(path: str) -> float:
    """File length in seconds from its header (NaN if unreadable)."""
    try:
        info = sf.info(path)
        return info.frames / info.samplerate
    except Exception:  # soundfile raises a range of errors on bad files
        return float("nan")


if __name__ == "__main__":
    # Dataset = the folder a file sits in, relative to its source root's parent,
    # e.g. dclde_2027_killer_whales/vfpa/audio/boundarypass or DORI-ONC/train_2015
    files = [(source, path, os.path.relpath(os.path.dirname(path), os.path.dirname(root)))
             for source, roots in AUDIO_DIRS.items()
             for root in ([roots] if isinstance(roots, str) else roots) for path in audio_files(root)]
    if Path(ONC_BACKGROUND_MANIFEST).exists():
        manifest = pd.read_csv(ONC_BACKGROUND_MANIFEST)
        files += [("onc_background", row.path, f"ONC-SoG-background/{row.site}")
                  for row in manifest.itertuples() if os.path.exists(row.path)]
    else:
        print(f"no ONC background manifest at {ONC_BACKGROUND_MANIFEST} -- building without it")
    files = pd.DataFrame(files, columns=["Source", "LocalPath", "Dataset"])

    with ThreadPoolExecutor(32) as pool:
        files["file_duration"] = list(tqdm(pool.map(duration, files["LocalPath"]), total=len(files), desc="reading headers", unit="file"))
    unreadable = files["file_duration"].isna()
    print(f"{len(files):,} files, {unreadable.sum()} unreadable (dropped), {files['file_duration'].sum() / 3600:,.0f} h")
    files = files[~unreadable]

    rng = np.random.default_rng(SEED)
    files["fold_0"] = np.where(rng.random(len(files)) < VAL_FRACTION, "val", "train")

    # non-overlapping windows; the last partial window of each file is dropped
    n_windows = (files["file_duration"] // WINDOW_SEC).astype(int)
    files = files[n_windows > 0]
    n_windows = n_windows[n_windows > 0]
    ssl = files.loc[files.index.repeat(n_windows)].reset_index(drop=True)
    ssl["FileBeginSec"] = np.concatenate([np.arange(n) * WINDOW_SEC for n in n_windows])
    ssl["FileEndSec"] = ssl["FileBeginSec"] + WINDOW_SEC
    ssl["Duration"] = WINDOW_SEC
    ssl["CenterTime"] = ssl["FileBeginSec"] + WINDOW_SEC / 2
    ssl["Labels"] = LABEL
    ssl["uid"] = np.arange(len(ssl))

    assert not ssl.duplicated(["LocalPath", "FileBeginSec"]).any()
    assert ssl.groupby("LocalPath")["fold_0"].nunique().max() == 1

    out_dir = Path(OUT_ROOT) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    columns = ["uid", "Source", "Dataset", "Labels", "LocalPath", "FileBeginSec", "FileEndSec", "Duration", "CenterTime"]
    ssl[columns].to_csv(out_dir / "annotations.csv", index=False)
    ssl[["uid", "fold_0"]].to_csv(out_dir / "splits.csv", index=False)
    print(pd.crosstab(ssl["Source"], ssl["fold_0"], margins=True).to_string())
    print(f"files: {ssl['LocalPath'].nunique():,}, hours: {len(ssl) * WINDOW_SEC / 3600:,.0f} -> {out_dir}")
