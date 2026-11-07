"""SSL annotations / splits over every *labelled* window: B's DCLDE clips + every DORI row that has any label.

B (build_ssl_source_annotations_v2.py) only takes DORI-ONC rows that are segment-level, human-generated and not
clicks (1,299 rows). This set takes every row of Bret's newer DORI.csv that has any label at all -- species, ecotype,
call or sighting annotation, human or pseudo-label -- from all four DORI folders, labels ignored (SSL):
  - DCLDE: B's DCLDE rows, unchanged, with B's train / val split (B's DORI rows are replaced by the rows below)
  - DORI segment rows (segment_start and segment_end): the segment, tiled into non-overlapping WINDOW_SEC windows
    (a segment shorter than one window gives one window centred on it)
  - DORI file-level rows (no segment_end -- mostly SanctSound presence labels on 5 min clips, segment_start = 0):
    the whole clip, tiled
  Overlapping rows of the same file are merged before tiling, so no window repeats. A file's last partial window is
  dropped.
Left out, as in B: the held-out VENUS deployments (Strait of Georgia / Saanich, within VENUS_RADIUS_M of a node; the
DCLDE ones -- LimeKiln, Tekteksen -- are already out of B), DORI files whose audio is also in DCLDE (same stem), DORI
files not on disk, and ONC-Benchmark (none of its files are in DORI.csv anyway).
val: VAL_FRACTION of the DORI files (grouped by file), B's val share.

usage:
    .venv/bin/python scripts/self_supervised/build_ssl_labelled_annotations.py
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

DORI_CSV = "/home/noah/ONC_download_project/DORI.csv"  # Bret's newer copy (HF still serves the older one)
DORI_DIRS = ["/data/DORI/DORI-ONC", "/data/DORI/DORI-OOI", "/data/DORI/DORI-Orcasound", "/data/DORI/DORI-SanctSound"]
B_DIR = "data_raw/ssl_dori_source/20261026_182504"  # B's SSL set: its DCLDE rows are reused as they are
DCLDE_ANNOTATIONS = "data_raw/DCLDE_2027/20260921_131510/annotations.csv"  # as build_DORI_annotations_v2.py
LABEL_COLUMNS = ["species_label", "species_label_clean", "ecotype_label_clean", "call_annotation",
                 "call_annotation_clean", "sighting_annotation"]
VENUS_NODES = [(49.042835, -123.317265), (49.040437, -123.425795), (48.650900, -123.486712)]  # SoG E, SoG C, Saanich
VENUS_RADIUS_M = 1000  # as build_DORI_annotations_v2.py
WINDOW_SEC = 3.0  # = 2 * preprocessor.annotation.time_offset
VAL_FRACTION = 0.157  # B's val share
OUT_ROOT = "data_raw/ssl_labelled"
COLUMNS = ["uid", "Source", "Dataset", "Labels", "LocalPath", "FileBeginSec", "FileEndSec", "Duration", "CenterTime"]
SEED = 10


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def duration(path: str) -> float:
    try:
        info = sf.info(path)
        return info.frames / info.samplerate
    except Exception:  # soundfile raises a range of errors on bad files
        return float("nan")


def merge_intervals(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def tile(start: float, end: float, file_duration: float) -> list[float]:
    """Window begin times covering [start, end), inside the file."""
    if end - start < WINDOW_SEC:  # short segment: one window centred on it
        begin = min(max(0.0, (start + end) / 2 - WINDOW_SEC / 2), file_duration - WINDOW_SEC)
        return [begin] if begin >= 0 else []
    return list(start + np.arange(int((end - start) // WINDOW_SEC)) * WINDOW_SEC)


if __name__ == "__main__":
    dori = pd.read_csv(DORI_CSV)
    dori["filename"] = dori["filename"].str.strip()
    labelled = dori[LABEL_COLUMNS].notna().any(axis=1)
    print(f"DORI.csv: {len(dori):,} rows, {labelled.sum():,} with a label")
    dori = dori[labelled]

    paths = {f: os.path.join(d, f) for root in DORI_DIRS for d, _, fs in os.walk(root) for f in fs
             if f.lower().endswith((".flac", ".wav"))}
    dori["LocalPath"] = dori["filename"].map(paths)
    dori["Dataset"] = dori["LocalPath"].str.split("/").str[3]  # DORI-ONC / DORI-OOI / ...
    near = np.min([haversine_m(dori["lat"], dori["lon"], lat, lon) for lat, lon in VENUS_NODES], axis=0)
    dclde_stems = set(pd.read_csv(DCLDE_ANNOTATIONS, usecols=["Soundfile"])["Soundfile"].dropna().map(lambda f: Path(f).stem))
    drops = {
        "not on disk": dori["LocalPath"].isna(),
        "held-out VENUS": pd.Series(near <= VENUS_RADIUS_M, index=dori.index),
        "audio also in DCLDE": dori["filename"].map(lambda f: Path(f).stem).isin(dclde_stems),
    }
    for name, mask in drops.items():
        print(f"  dropped, {name}: {mask.sum():,} rows")
    dori = dori[~pd.concat(drops, axis=1).any(axis=1)]

    files = dori.drop_duplicates("LocalPath")[["LocalPath", "Dataset"]].reset_index(drop=True)
    with ThreadPoolExecutor(32) as pool:
        files["file_duration"] = list(tqdm(pool.map(duration, files["LocalPath"]), total=len(files), desc="reading headers", unit="file"))
    print(f"  dropped, unreadable: {files['file_duration'].isna().sum()} files")
    files = files.dropna(subset=["file_duration"]).set_index("LocalPath")

    # per file: merged [start, end) intervals -- the whole clip if any of its rows is file-level
    rows = []
    for path, group in tqdm(dori[dori["LocalPath"].isin(files.index)].groupby("LocalPath"), desc="tiling", unit="file"):
        file_duration = files.at[path, "file_duration"]
        file_level = group["segment_end"].isna().any()
        intervals = [[0.0, file_duration]] if file_level else \
            [[max(0.0, s), min(e, file_duration)] for s, e in zip(group["segment_start"], group["segment_end"])]
        label = "|".join(sorted(group["species_label_clean"].combine_first(group["species_label"]).fillna("labelled").astype(str).unique()))
        for start, end in merge_intervals(intervals):
            for begin in tile(start, end, file_duration):
                rows.append((path, files.at[path, "Dataset"], label, "file" if file_level else "segment", begin))
    new = pd.DataFrame(rows, columns=["LocalPath", "Dataset", "Labels", "level", "FileBeginSec"])
    new["Source"] = "dori"

    rng = np.random.default_rng(SEED)
    val_files = set(files.index[rng.random(len(files)) < VAL_FRACTION])
    new["fold_0"] = np.where(new["LocalPath"].isin(val_files), "val", "train")

    b = pd.read_csv(f"{B_DIR}/annotations.csv", low_memory=False).merge(pd.read_csv(f"{B_DIR}/splits.csv"), on="uid")
    dclde = b[b["Source"] == "dclde"]
    print(f"B: {len(b):,} rows, {len(dclde):,} DCLDE kept, {len(b) - len(dclde):,} DORI replaced")

    new["FileEndSec"] = new["FileBeginSec"] + WINDOW_SEC
    new["Duration"] = WINDOW_SEC
    new["CenterTime"] = new["FileBeginSec"] + WINDOW_SEC / 2
    ssl = pd.concat([dclde[COLUMNS[1:] + ["fold_0"]], new[COLUMNS[1:] + ["fold_0"]]], ignore_index=True)
    ssl["uid"] = np.arange(len(ssl))

    assert not new.duplicated(["LocalPath", "FileBeginSec"]).any()
    assert ssl.groupby("LocalPath")["fold_0"].nunique().max() == 1

    out_dir = Path(OUT_ROOT) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    ssl[COLUMNS].to_csv(out_dir / "annotations.csv", index=False)
    ssl[["uid", "fold_0"]].to_csv(out_dir / "splits.csv", index=False)
    print(pd.crosstab(ssl["Source"], ssl["fold_0"], margins=True).to_string())
    print(pd.crosstab([new["Dataset"], new["level"]], new["fold_0"], margins=True).to_string())
    print(f"files: {ssl['LocalPath'].nunique():,}, hours: {len(ssl) * WINDOW_SEC / 3600:,.0f} (DCLDE rows count as 3 s) -> {out_dir}")
