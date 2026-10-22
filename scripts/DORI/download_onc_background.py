# Downloads background audio for the DORI v2 dataset from the ONC Strait of Georgia hydrophones.
# NOTE: The v1 'naive' background slid windows over DORI files, but DORI only labels one window per file,
# so those windows often contain unlabelled calls. Instead, whole files are pulled from the same ONC
# devices / deployments at times far from any DORI or DCLDE annotation (any label -- an annotation means
# something was there). Created from findings in NLP/EDA/onc_background.py.
#
# Needs an Oceans 3.0 API token (data.oceannetworks.ca -> Profile -> Web Services API) in $ONC_TOKEN.
# Set LIST_ONLY = True for a test run: queries ONC and writes the manifest, but downloads nothing.

import bisect
import csv
import datetime as dt
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

API = "https://data.oceannetworks.ca/api"
DORI_CSV = Path("/data/DORI/DORI-ONC/DORI.csv")
DCLDE_CSV = Path("/data/DCLDE_2027/dclde_2027_killer_whales/Annotations.csv")
# DCLDE datasets in the Strait of Georgia; their annotation times are excluded too
DCLDE_SOG_DATASETS = {"StraitofGeorgia", "StrGeoN1", "StrGeoN2", "StrGeoS1", "StrGeoS2"}
OUT_DIR = Path("/data/DORI/ONC-SoG-background")
# ONC Strait of Georgia sites DORI was recorded at: (lat, lon)
SITES = {
    "East": (49.04, -123.32),
    "Delta": (49.085, -123.335),
    "Central": (49.04, -123.43),
}
SITE_TOLERANCE_DEG = 0.02  # ~2 km
FILES_PER_SITE = 50
# DORI ran its detector over the ONC archive "through the end of 2022" (Nestor et al. 2026, arXiv 2602.09295) and
# kept only the high-likelihood data, so files it left out were screened with nothing found. Later audio was never
# screened, so deployments are clipped to this date.
DORI_SCAN_END = "2022-12-31T23:59:59.000Z"
MIN_HOURS_FROM_ANNOTATION = 24  # a file is only eligible this far from every SoG annotation (any device)
MIN_HOURS_BETWEEN_PICKS = 12  # spread picks over time (tide, season, vessel traffic)
# ONC serves different audio formats per device / period (some only high-pass filtered), so per timestamp the
# first of these that exists is used
AUDIO_SUFFIXES = [".flac", ".wav", "-HPF.flac", "-HPF.wav"]
SEED = 0
LIST_ONLY = False

TIMESTAMP = re.compile(r"^(.+?)_(\d{8}T\d{6})")


def site_of(lat, lon):
    for site, (site_lat, site_lon) in SITES.items():
        if abs(lat - site_lat) < SITE_TOLERANCE_DEG and abs(lon - site_lon) < SITE_TOLERANCE_DEG:
            return site


def parse_filename(filename):
    """(deviceCode, start time) from an ONC-style filename, e.g. ICLISTENHF1560_20200101T000000.000Z.flac"""
    match = TIMESTAMP.match(os.path.basename(filename))
    if not match:
        return None, None
    return match.group(1), dt.datetime.strptime(match.group(2), "%Y%m%dT%H%M%S")


def load_annotations():
    """SoG device codes from DORI, and sorted times of every DORI / DCLDE SoG annotation."""
    rows = list(csv.DictReader(open(DORI_CSV)))
    devices = {parse_filename(r["filename"])[0] for r in rows if r["lat"] and site_of(float(r["lat"]), float(r["lon"]))}
    devices.discard(None)
    # all rows of those devices, including ones without lat / lon
    times = []
    for r in rows:
        device, start = parse_filename(r["filename"])
        if device in devices:
            times.append(start + dt.timedelta(seconds=float(r["segment_start"] or 0)))
    n_dori = len(times)
    for r in csv.DictReader(open(DCLDE_CSV)):
        if r["Dataset"] in DCLDE_SOG_DATASETS and r["UTC"].strip():
            times.append(dt.datetime.fromisoformat(r["UTC"].strip()[:19]))
    print(f"{len(devices)} SoG devices in DORI; excluding around {n_dori} DORI + {len(times) - n_dori} DCLDE annotations")
    return sorted(devices), sorted(times)


def hours_to_nearest(t, times):
    i = bisect.bisect_left(times, t)
    return min(abs((times[j] - t).total_seconds()) for j in (i - 1, i) if 0 <= j < len(times)) / 3600


def onc_get(path, token, **params):
    url = f"{API}/{path}?" + urllib.parse.urlencode({**params, "token": token})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            if e.code == 404:  # e.g. no data for this device / window
                return None
            if e.code == 400:  # bad request -- surface ONC's error message rather than silently returning nothing
                raise RuntimeError(f"ONC rejected {path} {params}: {e.read().decode()}") from e
            if attempt == 2:
                raise
            time.sleep(5)


def list_audio_files(device, begin, end, token):
    """One audio filename per recording for a device in [begin, end] (following ONC's pagination),
    picking the first available format in AUDIO_SUFFIXES."""
    params, files = dict(deviceCode=device, dateFrom=begin, dateTo=end), []
    while params:
        result = onc_get("archivefile/device", token, **params) or {}
        files += result.get("files", [])
        params = (result.get("next") or {}).get("parameters")
        params = {k: v for k, v in params.items() if k != "token"} if params else None
    audio = re.compile(re.escape(device) + r"_(\d{8}T\d{6}\.\d{3}Z)(" + "|".join(map(re.escape, AUDIO_SUFFIXES)) + ")$")
    by_timestamp = defaultdict(dict)
    for f in files:
        match = audio.match(f)
        if match:
            by_timestamp[match.group(1)][match.group(2)] = f
    return [next(formats[s] for s in AUDIO_SUFFIXES if s in formats) for formats in by_timestamp.values()]


if __name__ == "__main__":
    token = os.environ.get("ONC_TOKEN")
    if not token:
        raise RuntimeError("set $ONC_TOKEN to an Oceans 3.0 API token")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    devices, annotation_times = load_annotations()

    # candidate files per site: audio files from SoG deployments, far from any annotation
    candidates = defaultdict(list)
    for device in devices:
        for deployment in onc_get("deployments", token, deviceCode=device) or []:
            site = site_of(deployment.get("lat") or 0, deployment.get("lon") or 0)
            if not site:
                continue
            # ISO strings in the same format, so min() / comparison is chronological
            end = min(deployment.get("end") or DORI_SCAN_END, DORI_SCAN_END)
            if deployment["begin"] >= end:
                continue
            files = list_audio_files(device, deployment["begin"], end, token)
            eligible = []
            for f in files:
                start = parse_filename(f)[1]
                gap = hours_to_nearest(start, annotation_times)
                if gap >= MIN_HOURS_FROM_ANNOTATION:
                    eligible.append((site, device, f, start, gap))
            print(f"  {site:8s} {device:28s} {deployment['begin'][:10]}..{end[:10]}  {len(files):6d} files, {len(eligible):6d} eligible")
            candidates[site] += eligible

    # shuffle, then greedily keep files >= MIN_HOURS_BETWEEN_PICKS from the ones already picked at that site
    rng, picks = random.Random(SEED), []
    for site, site_candidates in candidates.items():
        rng.shuffle(site_candidates)
        chosen = []
        for row in site_candidates:
            if all(abs((row[3] - p[3]).total_seconds()) >= MIN_HOURS_BETWEEN_PICKS * 3600 for p in chosen):
                chosen.append(row)
                if len(chosen) == FILES_PER_SITE:
                    break
        print(f"{site}: {len(site_candidates)} eligible -> picked {len(chosen)}")
        picks += chosen

    with open(OUT_DIR / "manifest.csv", "w", newline="") as fo:
        writer = csv.writer(fo)
        writer.writerow(["site", "device", "utc", "hours_to_nearest_annotation", "path"])
        for site, device, f, start, gap in sorted(picks, key=lambda r: (r[0], r[3])):
            writer.writerow([site, device, start.isoformat(), round(gap, 1), OUT_DIR / site / f])
    # params next to the manifest, so the v2 build can log what it was built from
    (OUT_DIR / "params.json").write_text(json.dumps({
        "sites": SITES, "site_tolerance_deg": SITE_TOLERANCE_DEG, "files_per_site": FILES_PER_SITE,
        "dori_scan_end": DORI_SCAN_END, "min_hours_from_annotation": MIN_HOURS_FROM_ANNOTATION, "min_hours_between_picks": MIN_HOURS_BETWEEN_PICKS,
        "audio_suffixes": AUDIO_SUFFIXES, "seed": SEED, "dori_csv": str(DORI_CSV), "dclde_csv": str(DCLDE_CSV),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
    }, indent=2))
    print(f"manifest -> {OUT_DIR / 'manifest.csv'}")
    if LIST_ONLY:
        raise SystemExit

    for i, (site, device, f, start, gap) in enumerate(picks, 1):
        dst = OUT_DIR / site / f
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"{API}/archivefile/download?" + urllib.parse.urlencode(dict(filename=f, token=token))
        urllib.request.urlretrieve(url, f"{dst}.part")
        os.replace(f"{dst}.part", dst)
        print(f"[{i}/{len(picks)}] {site} {f}")
