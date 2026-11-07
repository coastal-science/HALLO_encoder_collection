"""Leak-free file-level holdout for Strait of Georgia: files no model has trained on, and >= --pad-h hours away from
any Strait of Georgia training file on ANY hydrophone (consecutive 5-min ONC files, and the East / Central nodes, share
encounters).

File labels come from DORI-ONC's own human file-level labels (DORI.csv), not from our clip annotations: DORI is a
positive-unlabelled dataset, so an unannotated file is unreviewed, not negative.
  truth = SRKW / TKW / KW-unknown (species orca or 'multiple classes', by ecotype) or neg (humpback, Pacific white-sided
  dolphin, sea lion, noise, ...). Dropped: pseudo-labels, blank / 'uncertain' species.
Strait of Georgia = DORI.csv rows within 1 km of the East / Central VENUS nodes (as scripts/DORI/build_DORI_annotations_v1.py).

"Training file" = any file in
  * train / val of any fold of splits_pretrain_cmp_ft_<SoG>.csv (all fine-tuned arms and the Perch probe; EffNet's logged
    splits.csv is identical on every shared uid),
  * train / val of exp_2's source split (splits_bgcap10000.csv) on a SoG hydrophone (its holdout keys on the Dataset value,
    so DCLDE 'StraitofGeorgia' and some 'DORI ' files stayed in), EffNet-B1 / SSAMBA source pretraining,
  * DCLDE 2027 Dataset 'StraitofGeorgia' (in case an off-the-shelf model was trained on DCLDE).

    python scripts/adaptation/make_file_level_holdout.py [--pad-h 24]
writes scripts/adaptation/results/file_level/sog_holdout_<pad>h.csv: filename, LocalPath, truth, hydrophone, start_utc,
gap_h (hours to the nearest training file), species / ecotype labels.
"""
import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "data_raw/DORI/20261022_190549"
DORI_CSV = Path("/data/DORI/DORI-ONC/DORI.csv")
DORI_AUDIO = "/data/DORI/DORI-ONC/*/*"
DCLDE_CSV = REPO / "data_raw/DCLDE_2027/20260921_131510/annotations.csv"
SOG_NODES = [(49.042835, -123.317265), (49.040437, -123.425795)]  # East, Central
RADIUS_M = 1000


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


def parse_name(name: str) -> tuple[str | None, pd.Timestamp]:
    """ONC file name -> (hydrophone, start time)."""
    m = re.match(r"(.+?)_(\d{8}T\d{6})", os.path.basename(str(name)))
    return (m.group(1), pd.Timestamp(m.group(2))) if m else (None, pd.NaT)


def file_labels() -> pd.DataFrame:
    paths = {os.path.basename(p): p for p in glob.glob(DORI_AUDIO) if p.lower().endswith((".flac", ".wav"))}
    d = pd.read_csv(DORI_CSV, low_memory=False)
    d = d[d.filename.isin(paths)].dropna(subset=["lat", "lon"])
    dist = np.min([haversine_m(d.lat.round(3).values, d.lon.round(3).values, la, lo) for la, lo in SOG_NODES], axis=0)
    d = d[(dist <= RADIUS_M) & (d.species_label_source == "DORI (human-generated)")]
    join = lambda x: "|".join(sorted(set(map(str, x)) - {"nan"}))
    f = d.groupby("filename").agg(species=("species_label_clean", join), ecotype=("ecotype_label_clean", join))
    f = f[~f.species.isin(["", "uncertain"])]
    kw = f.species.str.contains("orca|multiple")
    f["truth"] = np.select([kw & f.ecotype.str.contains("srkw"), kw & (f.ecotype == "transient"), kw],
                           ["SRKW", "TKW", "KW-unknown"], "neg")
    f["LocalPath"] = f.index.map(paths)
    parsed = [parse_name(n) for n in f.index]
    f["hydrophone"], f["start_utc"] = [p[0] for p in parsed], [p[1] for p in parsed]
    return f


def training_files(sog_hydrophones: set[str]) -> set[str]:
    s = pd.read_csv(RUN / "splits_pretrain_cmp_ft_dori_venus_strait_of_georgia.csv")
    a = pd.read_csv(RUN / "annotations_bgcap10000_ft_dori_venus_strait_of_georgia.csv", low_memory=False, usecols=["uid", "LocalPath"])
    m = s.merge(a, on="uid")
    arms = set(m[m.filter(like="fold_").isin(["train", "val"]).any(axis=1)].LocalPath.map(os.path.basename))

    s = pd.read_csv(RUN / "splits_bgcap10000.csv")
    a = pd.read_csv(RUN / "annotations_bgcap10000.csv", low_memory=False, usecols=["uid", "LocalPath"])
    m = s.merge(a, on="uid")
    source = {os.path.basename(p) for p in m[m.filter(like="fold_").isin(["train", "val"]).any(axis=1)].LocalPath.unique()}
    source = {n for n in source if parse_name(n)[0] in sog_hydrophones}

    dc = pd.read_csv(DCLDE_CSV, low_memory=False, usecols=["Soundfile", "Dataset"])
    dclde = {os.path.basename(str(n)) for n in dc[dc.Dataset == "StraitofGeorgia"].Soundfile.unique()}
    print(f"training files: {len(arms)} fine-tune train/val, {len(source)} exp_2 source at SoG hydrophones, "
          f"{len(dclde)} DCLDE StraitofGeorgia")
    return arms | source | dclde


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pad-h", type=float, default=24.0)
    p.add_argument("--out-dir", type=Path, default=REPO / "scripts/adaptation/results/file_level")
    args = p.parse_args()

    f = file_labels()
    train = training_files(set(f.hydrophone.dropna()) | {"JASCOAMARHYDROPHONE2402", "JASCOAMARHYDROPHONEC000682",
                                                            "JASCOAMARHYDROPHONEB000882"})
    stems = {os.path.splitext(n)[0] for n in train}
    t_train = np.array(sorted(t for _, t in map(parse_name, train) if pd.notna(t)), dtype="datetime64[ns]")
    f = f[~f.index.map(lambda n: os.path.splitext(n)[0] in stems) & f.start_utc.notna()]
    starts = f.start_utc.values.astype("datetime64[ns]")
    idx = np.clip(np.searchsorted(t_train, starts), 1, len(t_train) - 1)
    gap = np.minimum(np.abs(starts - t_train[idx - 1]), np.abs(starts - t_train[idx])) / np.timedelta64(1, "h")
    f["gap_h"] = gap
    out = f[f.gap_h > args.pad_h].sort_values("start_utc")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"sog_holdout_{args.pad_h:g}h.csv"
    out.rename_axis("filename").reset_index()[
        ["filename", "LocalPath", "truth", "hydrophone", "start_utc", "gap_h", "species", "ecotype"]].to_csv(path, index=False)
    print(f"{len(out)} files ({out.truth.value_counts().to_dict()}) -> {path}")


if __name__ == "__main__":
    main()
