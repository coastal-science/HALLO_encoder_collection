import json
import csv
import hashlib
import requests
import argparse
from tqdm import tqdm
from pathlib import Path
from urllib.parse import quote

# ---------------------------
# Config
# ---------------------------
BASE_URL = "https://hearmyship.fer.hr/"
JSON_FILE = "vessels_hear_my_ship.json"
OUT_DIR = Path("data/")
INDEX_CSV = OUT_DIR / "hear_my_ship_index.csv"
ASSETS = ["sound", "image", "video"]

CHECKSUM = False   # set True if you want SHA-256 verification -> slow
CHUNK_SIZE = 1024 * 1024  # 1 MB
MAX_DOWNLOADS = None   # None = download everything

# ---------------------------
# Helpers
# ---------------------------
session = requests.Session()
session.headers.update({"User-Agent": "dataset-downloader/1.0"})

def count_assets(records):
    n = 0
    for rec in records:
        for key in ASSETS:
            if rec.get(key):
                n += 1
    return n

def normalize_path(p: str) -> str:
    """Convert Windows paths → POSIX and strip leading slashes."""
    if p is None:
        return None
    return p.replace("\\", "/").lstrip("/")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# -------------------------------------------------
# Index builder
# -------------------------------------------------
def build_index(records):
    rows = []

    for rec in records:
        rows.append({
            "id": rec.get("id"),
            "category": rec.get("category"),
            "subcategory": rec.get("subcategory"),
            "speed": rec.get("speed"),
            "length": rec.get("length"),
            "pressure": rec.get("pressure"),
            "time": rec.get("time"),
            "wav_path": normalize_path(rec.get("sound")),
            "image_path": normalize_path(rec.get("image")),
            "video_path": normalize_path(rec.get("video")),
        })

    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Index written: {INDEX_CSV}")

# -------------------------------------------------
# Downloader
# -------------------------------------------------
def download_file(rel_path: str):
    rel_path = normalize_path(rel_path)
    url = f"{BASE_URL}/{quote(rel_path, safe='()/%')}"

    dst = Path(OUT_DIR) / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)

    # HEAD request for integrity
    r_head = session.head(url, allow_redirects=True)
    r_head.raise_for_status()
    remote_size = int(r_head.headers.get("Content-Length", -1))

    # Resume / skip logic
    if dst.exists():
        if dst.stat().st_size == remote_size:
            return False # already downloaded
        else:
            print(f"[WARN] Size mismatch, re-downloading: {dst}")
            dst.unlink()

    # Download
    with session.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    # Integrity check
    if CHECKSUM:
        print(f"[CHECK] SHA256 {dst}: {sha256(dst)}")

    return True

def download_assets(records):
    total = count_assets(records)
    downloaded = 0

    with tqdm(total=total, unit="file", desc="Downloading assets") as pbar:
        for rec in records:
            for key in ASSETS:
                if MAX_DOWNLOADS is not None and downloaded >= MAX_DOWNLOADS:
                    print(f"[STOP] Reached MAX_DOWNLOADS={MAX_DOWNLOADS}")
                    return
                
                path = rec.get(key)
                if not path:
                    continue

                try:
                    did_download = download_file(path)
                    if did_download:
                        downloaded += 1
                        pbar.set_postfix(downloaded=downloaded)
                    pbar.update(1)
                except Exception as e:
                    pbar.update(1)
                    print(f"[ERROR] {path} → {e}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dataset-centric downloader for Hear My Ship"
    )

    parser.add_argument(
        "--json-file",
        type=str,
        default=JSON_FILE,
        help="Path to vessels_hear_my_ship.json",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for raw dataset",
    )

    parser.add_argument(
        "--max-downloads",
        type=int,
        default=MAX_DOWNLOADS,
        help="Maximum number of files to download (None = all)",
    )

    parser.add_argument(
        "--checksum",
        action="store_true",
        help="Enable SHA-256 checksum verification (slow)",
    )

    parser.add_argument(
        "--assets",
        nargs="+",
        default=ASSETS,
        choices=["sound", "image", "video"],
        help="Asset types to download",
    )

    return parser.parse_args()

def main():
    global OUT_DIR, JSON_FILE, MAX_DOWNLOADS, CHECKSUM, ASSETS, INDEX_CSV

    args = parse_args()

    OUT_DIR = args.out_dir
    JSON_FILE = args.json_file
    MAX_DOWNLOADS = args.max_downloads
    CHECKSUM = args.checksum
    ASSETS = args.assets
    INDEX_CSV = OUT_DIR / "hear_my_ship_index.csv"

    with open(JSON_FILE, "r") as f:
        records = json.load(f)

    build_index(records)

    download_assets(records)

if __name__ == "__main__":
    
    main()
