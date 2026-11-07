# Downloads the DORI audio that is not on this machine yet, into /data/DORI. Created from the comparison in
# /home/noah/ONC_download_project/compare_dori_csv.ipynb (sections 5 and 6).
#
#   hf    files in the DORI-SRKW Hugging Face repos that are missing locally (as of 2026-11-06: 491 DORI-OOI files,
#         added to HF on 2026-10-15, and 10 DORI-ONC files; ~12 GB). Saved with the repo layout, like the original pull.
#   noaa  DORI's SanctSound / ONMS / NRS files are 5 min clips that DORI cut from NOAA's ~6 h recordings, and no audio
#         for them is on HF. Each clip's parent recording (the latest one of the same deployment that starts before
#         it) is downloaded from the public NOAA bucket, the clips are cut out at (clip start - parent start) and
#         written at the parent's native rate / subtype, then the parent is deleted. ~0.44 TB downloaded for the
#         2026-11-05 DORI.csv (~0.91 TB with INCLUDE_SNAPSHOT); ~1-2 % of clips do not sit on the parent's 5 min grid
#         (off_grid in the plan) -- still cut at their timestamp, but check them.
# Wanted files that are on neither (some ONC / OOI files) are listed in the plan as "unavailable": they need the ONC
# API (see /home/noah/ONC_download_project/onc_pull.py) or OOI directly.
#
# Re-running skips files that already exist, so it can be stopped and restarted. Run with LIST_ONLY = True first: it
# lists HF and the NOAA bucket and writes the plan, but downloads nothing.
# Run with the project venv: .venv/bin/python scripts/DORI/download_dori_missing.py

import datetime as dt
import json
import re
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

# Bret's newer DORI.csv (not on HF yet: HF still serves the copy in /data/DORI/DORI-*/DORI.csv)
DORI_CSV = Path("/home/noah/ONC_download_project/DORI.csv")
SNAPSHOT_TXT = Path("/home/noah/ONC_download_project/marine_mammal_files_snapshot.txt")
INCLUDE_SNAPSHOT = False  # also cut the snapshot's NOAA clips that have no DORI label (~0.47 TB more)
DORI_ROOT = Path("/data/DORI")
EXTRA_LOCAL_ROOTS = [Path("/mnt/storage/data/onc_dataset")]  # files here count as local too
HF_ORG = "DORI-SRKW"
HF_REPOS = ["DORI-ONC", "DORI-OOI", "DORI-Orcasound", "ONC-Benchmark"]
NOAA_BUCKET = "noaa-passive-bioacoustic"
NOAA_OUT = DORI_ROOT / "DORI-SanctSound" / "data"
NOAA_PREFIXES = ("SanctSound", "ONMS", "NRS")
CLIP_SECONDS = 300
MIN_CLIP_SECONDS = 240  # a clip cut short by the end of its parent is kept ("short") if at least this long
PROGRESS_COLUMNS = ["time", "kind", "name", "status", "bytes", "parent", "offset_s", "off_grid", "seconds"]
MAX_PARENT_GAP_HOURS = 12  # parents are ~6 h; a clip further than this from any parent start is "no_parent"
TMP_DIR = DORI_ROOT / ".tmp_download"
LOG_DIR = DORI_ROOT / "logs" / "download_dori_missing"
STEPS = ("hf", "noaa")
LIST_ONLY = False
RETRIES = 5

GCS_API = f"https://storage.googleapis.com/storage/v1/b/{NOAA_BUCKET}/o"
GCS_FILES = f"https://storage.googleapis.com/{NOAA_BUCKET}/"
# clip / parent timestamps: 20190421T113327Z, 20150111_060516, 191018000711 (yymmddHHMMSS)
TIME_FORMATS = [(r"^(.*)_(\d{8}T\d{6})Z$", "%Y%m%dT%H%M%S"), (r"^(.*)_(\d{8}_\d{6})$", "%Y%m%d_%H%M%S"),
                (r"^(.*)_(\d{12})$", "%y%m%d%H%M%S")]


def log(msg):
    line = f"{dt.datetime.now():%F %T} {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "log.txt", "a") as f:
        f.write(line + "\n")


def record(kind, name, status, **extra):
    """Append one finished / failed item to progress.csv."""
    path = LOG_DIR / "progress.csv"
    row = pd.DataFrame([{"time": dt.datetime.now().isoformat(timespec="seconds"), "kind": kind, "name": name,
                         "status": status, **extra}], columns=PROGRESS_COLUMNS)
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def retry(fn, what):
    for attempt in range(1, RETRIES + 1):
        try:
            return fn()
        except Exception as e:  # network errors, truncated downloads
            if attempt == RETRIES:
                raise
            log(f"  retry {attempt}/{RETRIES} {what}: {e!r}")
            time.sleep(10 * attempt)


def parse_time(filename):
    """(deployment, start time) from a NOAA-style filename, or (None, None)."""
    stem = filename.rsplit(".", 1)[0]
    for pattern, fmt in TIME_FORMATS:
        m = re.match(pattern, stem)
        if m:
            try:
                return m.group(1).upper(), dt.datetime.strptime(m.group(2), fmt)
            except ValueError:  # digits that are not a valid time (some bucket names)
                pass
    return None, None


def gcs_prefix(deployment):
    """Bucket folder holding a deployment's recordings."""
    if m := re.match(r"SANCTSOUND_([A-Z]+\d+)_", deployment):
        return f"sanctsound/audio/{m.group(1).lower()}/"
    if m := re.match(r"ONMS_([A-Z]+\d+)_", deployment):
        return f"onms/audio/{m.group(1).lower()}/"
    if m := re.match(r"NRS(\d+)$", deployment):
        return f"nrs/audio/{m.group(1)}/"


def gcs_list(prefix):
    out, token = [], None
    while True:
        q = {"prefix": prefix, "maxResults": 1000, "fields": "items(name,size),nextPageToken"}
        if token:
            q["pageToken"] = token
        d = retry(lambda: json.load(urllib.request.urlopen(GCS_API + "?" + urllib.parse.urlencode(q), timeout=60)),
                  prefix)
        out += [(i["name"], int(i["size"])) for i in d.get("items", []) if i["name"].endswith(".flac")]
        token = d.get("nextPageToken")
        if not token:
            return out


def local_names():
    names = set()
    for root in [DORI_ROOT, *EXTRA_LOCAL_ROOTS]:
        names |= {p.name for p in root.rglob("*.flac")}
    return names


def plan_hf(have):
    rows = []
    api = HfApi()
    for repo in HF_REPOS:
        for f in api.list_repo_tree(f"{HF_ORG}/{repo}", repo_type="dataset", recursive=True):
            if hasattr(f, "size") and f.path.endswith(".flac") and Path(f.path).name not in have:
                rows.append({"kind": "hf", "name": Path(f.path).name, "repo": repo, "path": f.path, "bytes": f.size})
    return pd.DataFrame(rows, columns=["kind", "name", "repo", "path", "bytes"])


def plan_noaa(wanted):
    clips = pd.DataFrame({"name": sorted(n for n in wanted if n.startswith(NOAA_PREFIXES))})
    clips[["deployment", "t"]] = [parse_time(n) for n in clips.name]
    listing = []
    for prefix in sorted({gcs_prefix(d) for d in clips.deployment.dropna()} - {None}):
        listing += gcs_list(prefix)
        log(f"listed gs://{NOAA_BUCKET}/{prefix}: {len(listing)} recordings so far")
    parents = pd.DataFrame(listing, columns=["parent_path", "parent_bytes"])
    parents["parent"] = parents.parent_path.str.rsplit("/", n=1).str[-1]
    parents[["deployment", "parent_t"]] = [parse_time(n) for n in parents.parent]
    parents.to_csv(LOG_DIR / "noaa_listing.csv", index=False)
    m = pd.merge_asof(clips.dropna(subset=["t"]).sort_values("t"),
                      parents.dropna(subset=["parent_t"]).sort_values("parent_t"),
                      left_on="t", right_on="parent_t", by="deployment", direction="backward")
    m = pd.concat([m, clips[clips.t.isna()]])  # unparseable names -> no_parent
    m["offset_s"] = (m.t - m.parent_t).dt.total_seconds()
    m["off_grid"] = m.offset_s.notna() & (m.offset_s % CLIP_SECONDS != 0)
    m["kind"] = "noaa"
    m.loc[m.parent.isna() | (m.offset_s > MAX_PARENT_GAP_HOURS * 3600), "kind"] = "no_parent"
    return m[["kind", "name", "deployment", "parent", "parent_path", "parent_bytes", "offset_s", "off_grid"]]


def run_hf(plan):
    for i, r in enumerate(plan.itertuples(), 1):
        if (DORI_ROOT / r.repo / r.path).exists():
            continue
        retry(lambda: hf_hub_download(f"{HF_ORG}/{r.repo}", r.path, repo_type="dataset", local_dir=DORI_ROOT / r.repo),
              r.path)
        record("hf", r.name, "done", bytes=r.bytes)
        log(f"hf {i}/{len(plan)} {r.repo}/{r.path}")


def download(url, dest):
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(part, "wb") as f:
        shutil.copyfileobj(resp, f, 1 << 20)
    part.rename(dest)


def run_noaa(plan):
    NOAA_OUT.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    # parents with the most clips first, so a partial run covers as many clips per byte as possible
    groups = sorted(plan.groupby("parent_path"), key=lambda g: -len(g[1]))
    done_bytes, total_bytes = 0, plan.drop_duplicates("parent_path").parent_bytes.sum()
    for i, (path, clips) in enumerate(groups, 1):
        clips = clips[[not (NOAA_OUT / n).exists() for n in clips.name]]
        size = clips.parent_bytes.iloc[0] if len(clips) else 0
        if clips.empty:
            continue
        parent = TMP_DIR / Path(path).name
        retry(lambda: download(GCS_FILES + path, parent), path)
        with sf.SoundFile(parent) as src:
            for c in clips.itertuples():
                start, n = int(round(c.offset_s * src.samplerate)), CLIP_SECONDS * src.samplerate
                # parents are ~3 s short of 6 h, so a clip in the last 5 min slot runs past the end: keep it short
                n = min(n, src.frames - start)
                if n < MIN_CLIP_SECONDS * src.samplerate:
                    record("noaa", c.name, "beyond_parent", parent=c.parent, offset_s=c.offset_s)
                    continue
                src.seek(start)
                out = NOAA_OUT / c.name
                tmp = out.with_name(out.name + ".part")
                sf.write(tmp, src.read(n, dtype="int32" if "24" in src.subtype else "int16"), src.samplerate,
                         subtype=src.subtype, format="FLAC")
                tmp.rename(out)
                status = "done" if n == CLIP_SECONDS * src.samplerate else "short"
                record("noaa", c.name, status, parent=c.parent, offset_s=c.offset_s, off_grid=c.off_grid,
                       seconds=n / src.samplerate)
        parent.unlink()
        done_bytes += size
        log(f"noaa {i}/{len(groups)} {Path(path).name}: {len(clips)} clips, "
            f"{done_bytes / 1e9:.0f}/{total_bytes / 1e9:.0f} GB")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    wanted = set(pd.read_csv(DORI_CSV, usecols=["filename"]).filename)
    if INCLUDE_SNAPSHOT:
        wanted |= {l.strip() for l in open(SNAPSHOT_TXT) if l.strip()}
    have = local_names()
    log(f"start: {len(wanted)} wanted files, {len(have)} local; steps={STEPS} list_only={LIST_ONLY}")

    hf = plan_hf(have) if "hf" in STEPS else pd.DataFrame(columns=["kind", "name"])
    noaa = plan_noaa(wanted - have) if "noaa" in STEPS else pd.DataFrame(columns=["kind", "name"])
    covered = have | set(hf.name) | set(noaa.name)
    unavailable = pd.DataFrame({"kind": "unavailable", "name": sorted(wanted - covered)})
    plan = pd.concat([hf, noaa, unavailable], ignore_index=True)
    plan.to_csv(LOG_DIR / "plan.csv", index=False)

    noaa_ok = plan[plan.kind == "noaa"]
    log(f"plan -> {LOG_DIR / 'plan.csv'}: {plan.kind.value_counts().to_dict()}")
    log(f"  hf: {hf.bytes.sum() / 1e9:.1f} GB; noaa: {noaa_ok.parent.nunique()} parents, "
        f"{noaa_ok.drop_duplicates('parent').parent_bytes.sum() / 1e12:.2f} TB to download, "
        f"{int(noaa_ok.off_grid.sum())} off-grid clips")
    if LIST_ONLY:
        return
    if "hf" in STEPS:
        run_hf(hf)
    if "noaa" in STEPS:
        run_noaa(noaa_ok)
    log("done")


if __name__ == "__main__":
    main()
