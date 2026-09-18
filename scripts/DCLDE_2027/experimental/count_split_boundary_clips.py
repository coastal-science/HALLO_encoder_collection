"""Count clips that are split across the train/test boundary and share a label at the seam.

A hit is a pair of consecutive recordings (A, then B) on the same instrument where

  - A and B were recorded back-to-back: start(B) - start(A) == length(A)  (within --gap-tol);
  - A is in one split and B is in the other (one train, one test);
  - A has a non-Background annotation of class C ending at A's end   (within --edge-tol);
  - B has a non-Background annotation of the same class C starting at B's start (within --edge-tol).

That last-annotation-of-A / first-annotation-of-B pair is taken to be one acoustic
instance cut by the file boundary.

Prints the count (total, by dataset, by class, by train->test / test->train direction) and writes:
  --out                every involved annotations.csv row, original columns plus
                       fold_0, seam_role, seam_class, seam_partner_soundfiles
  --exclude-train-out  splits-style uid,fold_0 CSV of the train-side rows on a seam
  --exclude-test-out   splits-style uid,fold_0 CSV of the test-side rows on a seam
                       Drop these so no call keeps one half in train and the other in
                       the holdout. --exclude-scope file (default) lists every row of
                       each clip that touches a seam; 'annotation' lists only the
                       boundary annotations themselves.

Usage:
    python count_split_boundary_clips.py [--edge-tol 0.6] [--gap-tol 6.0] [--out split_boundary_clips.csv]
        [--exclude-train-out exclude_train.csv] [--exclude-test-out exclude_test.csv] [--exclude-scope file|annotation]
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = Path("/home/noah/HALLO_encoder_collection/data_raw/DCLDE_2027/20260827_090049")
ANNOTATIONS = BASE / "annotations.csv"
SPLITS = BASE / "reconstructed_splits" / "splits_full_train.csv"


def parse_start(name: str):
    """Recording start time embedded in the audio filename (UTC)."""
    s = re.sub(r"\.(wav|flac)$", "", str(name), flags=re.I)

    def dt(*a, frac=0.0):
        try:
            return datetime(*a, int(frac * 1e6), tzinfo=timezone.utc)
        except (ValueError, OverflowError):
            return None

    m = re.search(r"(19|20)(\d\d)(\d\d)(\d\d)[_T](\d\d)(\d\d)(\d\d)(\.\d+)?", s)
    if m:
        c, y, mo, d, h, mi, se, fr = m.groups()
        return dt(int(c + y), int(mo), int(d), int(h), int(mi), int(se), frac=float(fr or 0))
    m = re.search(r"(19|20)(\d\d)(\d\d)(\d\d)[_T](\d\d)(\d\d)(?!\d)", s)
    if m:
        c, y, mo, d, h, mi = m.groups()
        return dt(int(c + y), int(mo), int(d), int(h), int(mi))
    m = re.search(r"((?:19|20)\d\d)_(\d\d)_(\d\d)_(\d\d)_(\d\d)_(\d\d)", s)
    if m:
        yr, mo, d, h, mi, se = m.groups()
        return dt(int(yr), int(mo), int(d), int(h), int(mi), int(se))
    m = re.search(r"(?<!\d)(\d{1,2})_(\d{1,2})_((?:19|20)\d\d)_(\d\d)_(\d\d)_(\d\d)", s)
    if m:
        mo, d, yr, h, mi, se = m.groups()
        return dt(int(yr), int(mo), int(d), int(h), int(mi), int(se))
    m = re.search(r"(?<!\d)(\d\d)(\d\d)(\d\d)_(\d\d)(\d\d)(\d\d)(?!\d)", s)
    if m:
        y, mo, d, h, mi, se = m.groups()
        return dt(2000 + int(y), int(mo), int(d), int(h), int(mi), int(se))
    m = re.search(r"(?<!\d)(\d\d)(\d\d)(\d\d)(\d\d)(\d\d)(\d\d)(?!\d)", s)
    if m:
        y, mo, d, h, mi, se = m.groups()
        return dt(2000 + int(y), int(mo), int(d), int(h), int(mi), int(se))
    m = re.search(r"(?<!\d)(1[0-9]{9})(?!\d)", s)
    if m:
        try:
            return datetime.fromtimestamp(int(m.group(1)), tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None
    return None


def instrument_key(name: str) -> str:
    """Filename prefix before the embedded timestamp."""
    s = re.sub(r"\.(wav|flac)$", "", str(name), flags=re.I)
    m = re.search(r"[._T -]?((19|20)\d{2})[._T0-9-]*$", s)
    if m:
        return s[: m.start()].rstrip("._-T ") or name
    m = re.search(r"[._-]?\d{6,}", s)
    if m:
        return s[: m.start()].rstrip("._-T ") or name
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    ap.add_argument("--splits", type=Path, default=SPLITS)
    ap.add_argument("--edge-tol", type=float, default=0.6,
                    help="seconds an annotation may sit from the seam and still 'touch' it")
    ap.add_argument("--gap-tol", type=float, default=6.0,
                    help="seconds start(B)-start(A) may differ from length(A) and still be back-to-back")
    ap.add_argument("--out", type=Path, default=Path("split_boundary_clips.csv"),
                    help="CSV to write with every annotations.csv row that sits on such a seam")
    ap.add_argument("--exclude-train-out", type=Path, default=Path("exclude_train.csv"),
                    help="splits-style uid,fold_0 CSV of the train-side rows on a seam")
    ap.add_argument("--exclude-test-out", type=Path, default=Path("exclude_test.csv"),
                    help="splits-style uid,fold_0 CSV of the test-side rows on a seam")
    ap.add_argument("--exclude-scope", choices=("file", "annotation"), default="file",
                    help="'file': list every row of each clip touching a seam; "
                         "'annotation': list only the boundary annotations themselves")
    args = ap.parse_args()

    a = pd.read_csv(
        args.annotations, low_memory=False,
        usecols=["uid", "Soundfile", "Dataset", "FileBeginSec", "FileEndSec", "Labels"],
    )
    a["FileBeginSec"] = pd.to_numeric(a["FileBeginSec"], errors="coerce")
    a["FileEndSec"] = pd.to_numeric(a["FileEndSec"], errors="coerce")

    s = pd.read_csv(args.splits)                       # uid, fold_0 in {train, test}
    df = a.merge(s, on="uid", how="inner")

    flen = df.groupby("Soundfile")["FileEndSec"].max()    # file length ~= last annotation end

    files = df.groupby("Soundfile", as_index=False).agg(Dataset=("Dataset", "first"))
    files["split"] = df.groupby("Soundfile")["fold_0"].agg(lambda x: x.mode().iat[0]).values
    files["start"] = pd.to_datetime(files["Soundfile"].map(parse_start), utc=True)
    files["len"] = files["Soundfile"].map(flen)
    files["inst"] = files["Dataset"] + " / " + files["Soundfile"].map(instrument_key)
    files = files.dropna(subset=["start"]).sort_values("start")

    ev = df[df["Labels"] != "Background"].copy()
    ev["flen"] = ev["Soundfile"].map(flen)
    tail_rows = ev[ev["FileEndSec"] >= ev["flen"] - args.edge_tol]      # annotation ends at file end
    head_rows = ev[ev["FileBeginSec"] <= args.edge_tol]                 # annotation starts at file start
    tail_uid = {sf: g.groupby("Labels")["uid"].agg(list) for sf, g in tail_rows.groupby("Soundfile")}
    head_uid = {sf: g.groupby("Labels")["uid"].agg(list) for sf, g in head_rows.groupby("Soundfile")}

    hits: list[dict] = []          # one per (file pair, shared class)
    for _, g in files.groupby("inst", sort=False):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            A, B = g.loc[i], g.loc[i + 1]
            if A["split"] == B["split"]:
                continue
            if abs((B["start"] - A["start"]).total_seconds() - A["len"]) > args.gap_tol:
                continue                                               # not back-to-back
            ta, hb = tail_uid.get(A["Soundfile"]), head_uid.get(B["Soundfile"])
            if ta is None or hb is None:
                continue
            for cls in set(ta.index) & set(hb.index):
                hits.append({
                    "hit_id": len(hits),
                    "dataset": A["Dataset"],
                    "class": cls,
                    "file_ends": A["Soundfile"], "split_ends": A["split"],
                    "file_starts": B["Soundfile"], "split_starts": B["split"],
                    "uids_ends": ta[cls], "uids_starts": hb[cls],
                })

    # ---- console summary -------------------------------------------------------
    clips = len(hits)
    print(f"clips split across the train/test boundary that share a label at the seam: {clips}")
    print()

    def tally(key):
        out: dict = {}
        for h in hits:
            out[h[key]] = out.get(h[key], 0) + 1
        for k, v in sorted(out.items(), key=lambda kv: -kv[1]):
            print(f"  {str(k):<16} {v}")

    print("by dataset:"); tally("dataset")
    print("by class:"); tally("class")
    print("by direction:")
    print(f"  train -> test    {sum(h['split_ends'] == 'train' for h in hits)}")
    print(f"  test  -> train   {sum(h['split_ends'] == 'test'  for h in hits)}")

    # ---- CSV of every annotation row sitting on such a seam -------------------
    role = {}          # uid -> set of roles
    partners = {}       # uid -> set of files on the other side
    classes = {}        # uid -> set of seam classes
    for h in hits:
        for u in h["uids_ends"]:
            role.setdefault(u, set()).add("ends_at_seam")
            partners.setdefault(u, set()).add(h["file_starts"])
            classes.setdefault(u, set()).add(h["class"])
        for u in h["uids_starts"]:
            role.setdefault(u, set()).add("starts_at_seam")
            partners.setdefault(u, set()).add(h["file_ends"])
            classes.setdefault(u, set()).add(h["class"])

    full = pd.read_csv(args.annotations, low_memory=False)
    involved = full[full["uid"].isin(role)].copy()
    fold = df.set_index("uid")["fold_0"]
    involved["fold_0"] = involved["uid"].map(fold)
    involved["seam_role"] = involved["uid"].map(lambda u: "+".join(sorted(role[u])))
    involved["seam_class"] = involved["uid"].map(lambda u: ";".join(sorted(classes[u])))
    involved["seam_partner_soundfiles"] = involved["uid"].map(lambda u: ";".join(sorted(partners[u])))
    involved.to_csv(args.out, index=False)

    n_train = int((involved["fold_0"] == "train").sum())
    n_test = int((involved["fold_0"] == "test").sum())
    print()
    print(f"wrote {len(involved):,} annotation rows to {args.out}")
    print(f"  in train : {n_train}")
    print(f"  in test  : {n_test}")

    # ---- splits-style exclusion lists, one per split ----------------------
    # Every seam has one train clip and one test clip. These two files list
    # the rows to drop from each split so no call keeps one half in train
    # and the other in the holdout. No clip is mixed, so each list stays
    # entirely within its own split.
    def seam_clips(split):
        return {h["file_ends"] if h["split_ends"] == split else h["file_starts"] for h in hits}

    def seam_boundary_uids(split):
        u: set = set()
        for h in hits:
            u.update(h["uids_ends"] if h["split_ends"] == split else h["uids_starts"])
        return u

    for split, out_path in (("train", args.exclude_train_out), ("test", args.exclude_test_out)):
        clips = seam_clips(split)
        if args.exclude_scope == "file":
            sel = df[df["Soundfile"].isin(clips)]
        else:
            sel = df[df["uid"].isin(seam_boundary_uids(split))]
        sel = sel[sel["fold_0"] == split][["uid", "fold_0"]].sort_values("uid")
        sel.to_csv(out_path, index=False)
        print()
        print(f"wrote {len(sel):,} {split} uids ({args.exclude_scope} scope) to {out_path}")
        print(f"  from {len(clips)} {split} clips across {len(hits)} seams")


if __name__ == "__main__":
    main()
