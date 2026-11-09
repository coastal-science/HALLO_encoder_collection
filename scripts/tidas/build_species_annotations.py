# TIDAS species-vs-all probe: the human-reviewed annotations, as 5 s windows for a Perch v2 linear probe
# (configs/tidas/01_perch_v2_species_probe_5fold.yaml). The probe then labels the unscreened audio the ONC / NOAA /
# OOI pull puts on the NAS (/mnt/nas/TIDAS), to find new positives for SSL pretraining.
#
# Sources (human-reviewed only):
#   DCLDE 2027   data_raw/DCLDE_2027/<run>/annotations.csv -- analyst call boxes (KW ecotypes, HW, AB = abiotic,
#                UndBio) plus its Background windows. KW ecotypes -> orca, HW -> humpback, AB -> Background, UndBio
#                dropped (unknown biology: neither a species nor clean background).
#   DORI         ONC_download_project/DORI.csv rows whose species_label_source is human: "DORI (human-generated)"
#                and "SanctSound" (NOAA analyst detections). Each row marks a 15 s segment (segment_start / _end) in
#                a 5 min clip; the segment is tiled into 5 s windows. "DORI (pseudo-label)" / "ONC" rows are dropped.
#   ONC SoG bg   the Background windows DORI v2 tiled from ONC Strait of Georgia files far from any annotation.
# Rows are capped (per class, per file) so the CPU-only Perch embedding stays a few hours; caps are below.
# Run: /home/noah/embed_venv/bin/python scripts/tidas/build_species_annotations.py

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DCLDE_CSV = REPO / "data_raw/DCLDE_2027/20260921_131510/annotations.csv"
DORI_CSV = Path("/home/noah/ONC_download_project/DORI.csv")  # Bret's newer DORI.csv (59,352 of its files on disk)
DORI_V2_CSV = REPO / "data_raw/DORI/20261022_190549/annotations.csv"  # for the ONC SoG background windows
DORI_ROOT = Path("/data/DORI")
OUT_ROOT = REPO / "data_raw/tidas"

HUMAN_SOURCES = ["DORI (human-generated)", "SanctSound"]
WINDOW_S = 5.0  # Perch v2's window
SEGMENT_S = 15.0  # DORI's labelled segment length, used when segment_end is missing
SPECIES = {  # DORI species_label_clean -> probe class
    "orca": "orca", "humpback": "humpback", "fin whale": "fin_whale",
    "pacific white sided dolphin": "pacific_white_sided_dolphin", "sea lion": "sea_lion",
    "false killer whale": "other_marine_mammal", "sperm whale": "other_marine_mammal", "gray": "other_marine_mammal",
    "minke": "other_marine_mammal", "northern right whale dolphin": "other_marine_mammal", "noise": "Background",
}
DCLDE_MAP = {"SRKW": "orca", "TKW": "orca", "NRKW": "orca", "OKW": "orca", "SAR": "orca", "KW_und": "orca",
             "HW": "humpback", "AB": "Background", "Background": "Background"}
MAX_PER_CLASS = 20_000  # windows per species class (fewer when the class is smaller)
MAX_BACKGROUND = 40_000  # spread evenly over the background sources' datasets
MAX_PER_FILE = 4  # windows per audio file, so a class is not a few long encounters
SEED = 0


def dori_local_paths() -> dict[str, str]:
    return {p.name: str(p) for p in DORI_ROOT.rglob("*.flac") if "ONC-SoG-background" not in p.parts}


def dclde_rows() -> pd.DataFrame:
    d = pd.read_csv(DCLDE_CSV, low_memory=False)
    d = d[d.Labels.isin(DCLDE_MAP) & (d.Duration > 0) & d.FileBeginSec.notna()].copy()
    d["Labels"] = d.Labels.map(DCLDE_MAP)
    d["Source"], d["LabelSource"] = "DCLDE_2027", "DCLDE 2027 (analyst)"
    return d[["LocalPath", "FileBeginSec", "Duration", "Labels", "Dataset", "Source", "LabelSource"]]


def dori_rows() -> pd.DataFrame:
    d = pd.read_csv(DORI_CSV, low_memory=False)
    # most SanctSound rows (and some DORI ones) carry only segment_start: DORI's segments are 15 s
    d["segment_end"] = d.segment_end.fillna(d.segment_start + SEGMENT_S)
    d = d[d.species_label_source.isin(HUMAN_SOURCES) & d.species_label_clean.isin(SPECIES)
          & d.segment_start.notna()].copy()
    paths = dori_local_paths()
    d["LocalPath"] = d.filename.map(paths)
    print(f"DORI human rows {len(d)}, {d.LocalPath.isna().sum()} without a local file (dropped)")
    d = d[d.LocalPath.notna()]
    rows = []
    for r in d.itertuples():
        n = max(1, int((r.segment_end - r.segment_start) // WINDOW_S))
        for k in range(n):  # tile the labelled segment into back-to-back 5 s windows
            rows.append((r.LocalPath, r.segment_start + k * WINDOW_S, WINDOW_S, SPECIES[r.species_label_clean],
                         str(r.LocalPath).split("/")[3], "DORI", r.species_label_source))
    return pd.DataFrame(rows, columns=["LocalPath", "FileBeginSec", "Duration", "Labels", "Dataset", "Source",
                                       "LabelSource"])


def onc_sog_background() -> pd.DataFrame:
    d = pd.read_csv(DORI_V2_CSV, low_memory=False)
    d = d[(d.Labels == "Background") & d.LocalPath.str.contains("ONC-SoG-background", na=False)].copy()
    d["Source"], d["LabelSource"] = "ONC_SoG_background", "far from any DORI / DCLDE annotation"
    return d[["LocalPath", "FileBeginSec", "Duration", "Labels", "Dataset", "Source", "LabelSource"]]


def cap(df: pd.DataFrame, n: int, rng: np.random.Generator, by: str | None = None) -> pd.DataFrame:
    """At most MAX_PER_FILE rows per file, then n rows (spread evenly over `by` groups when given)."""
    df = df.sample(frac=1, random_state=int(rng.integers(1 << 31)))
    df = df.groupby("LocalPath", sort=False).head(MAX_PER_FILE)
    if by is None:
        return df.head(n)
    groups = [g for _, g in df.groupby(by)]
    out, quota = [], n
    for i, g in enumerate(sorted(groups, key=len)):  # small groups take all they have, the rest share what is left
        take = quota // (len(groups) - i)
        out.append(g.head(take))
        quota -= min(take, len(g))
    return pd.concat(out)


def main() -> None:
    rng = np.random.default_rng(SEED)
    pool = pd.concat([dclde_rows(), dori_rows(), onc_sog_background()], ignore_index=True)
    print("pool (windows) by class and source:\n", pd.crosstab(pool.Labels, pool.Source))
    parts = [cap(pool[pool.Labels == "Background"], MAX_BACKGROUND, rng, by="Dataset")]
    for label in sorted(set(pool.Labels) - {"Background"}):
        parts.append(cap(pool[pool.Labels == label], MAX_PER_CLASS, rng))
    df = pd.concat(parts, ignore_index=True)
    df["FileEndSec"] = df.FileBeginSec + df.Duration
    df["CenterTime"] = df.FileBeginSec + df.Duration / 2
    df.insert(0, "uid", np.arange(len(df)))
    out_dir = OUT_ROOT / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    df.to_csv(out_dir / "annotations_species.csv", index=False)
    summary = {
        "classes": pd.crosstab(df.Labels, df.Source).to_dict(), "files": int(df.LocalPath.nunique()),
        "windows": len(df), "caps": {"per_class": MAX_PER_CLASS, "background": MAX_BACKGROUND, "per_file": MAX_PER_FILE},
        "human_sources": HUMAN_SOURCES, "dclde_map": DCLDE_MAP, "dori_map": SPECIES, "seed": SEED,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nselected windows by class and source:\n", pd.crosstab(df.Labels, df.Source, margins=True))
    print(f"\n{len(df)} windows on {df.LocalPath.nunique()} files -> {out_dir}/annotations_species.csv")


if __name__ == "__main__":
    main()
