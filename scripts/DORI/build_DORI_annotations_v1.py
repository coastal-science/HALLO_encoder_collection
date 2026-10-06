# HALLO Encoder Collection DORI integration v1 script 
# NOTE: This script is created from findings based in the noah/HALLO_encoder_collection/scripts/DORI/experimental/eda.ipynb file

import subprocess
import sys
import time
from pathlib import Path

import mlflow

from encoder_pipeline.common.mlflow_utils import configure_datasets_mlflow

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

DCLDE_PATH = "/home/noah/HALLO_encoder_collection/data_raw/DCLDE_2027/20260921_131510/annotations.csv"
VERIFY_HF = False
DATASET_NAME = "DORI"
DORI_ROOT = Path("/data/DORI")
DATA_DIR = REPO_ROOT / "data_raw" / DATASET_NAME
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = f"Datasets/{DATASET_NAME}"
# How to select 'Background' windows; only 'naive' is implemented so far.
BACKGROUND_METHOD = "naive"
BACKGROUND_WINDOW_DURATION = 5.0
BACKGROUND_HOP_DURATION = 2.5
BACKGROUND_EVENT_BUFFER = 10.0
BACKGROUND_LABEL = "Background"


def git_commit(repo_dir: Path) -> str:
    """Current commit hash of the repo this script lives in."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True,
    ).stdout.strip()


def check_uncommitted_changes(repo_dir: Path) -> None:
    """Raises if the working tree has uncommitted changes -- dataset generation
    requires a clean tree so git_commit fully captures what actually ran
    (including the PARAMS above, which only live in this file)."""
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True, check=True,
    ).stdout
    if status.strip():
        raise RuntimeError(f"uncommitted changes in {repo_dir} -- commit before running:\n{status}")


if __name__ == "__main__":
    import pandas as pd

    check_uncommitted_changes(SCRIPT_DIR)
    run_dir = DATA_DIR / time.strftime("%Y%m%d_%H%M%S")

    # Note: each csv contains the same information, this is loading in the entire DORI dataset csv 
    DORI_ONC = pd.read_csv(DORI_ROOT / "DORI-ONC" / "DORI.csv")
    print(DORI_ONC["species_label_clean"].value_counts())
    print(DORI_ONC["ecotype_label_clean"].value_counts())
    print(DORI_ONC["call_annotation_clean"].value_counts())
    # step 1: 
    # In its current form, not all data is available on the huggingface DS:
    # DORI-ONC: on_disk=32247 missing=34891 (58.7%)
    # DORI-OOI: on_disk=1356 missing=59090 (99.4%)
    # DORI-Orcasound: on_disk=1585 missing=57665 (97.0%)
    # DORI-SanctSound: on_disk=0 missing=59442 (100.0%)
    # Check which files listed in the shared DORI.csv manifest are missing on disk,
    # broken down per repo (the manifest is identical across all four DORI-* repos,
    # but the audio is split across their individual data/train_*/test folders)
    DATA_DIRS = [
        DORI_ROOT / "DORI-ONC",
        DORI_ROOT / "DORI-OOI",
        DORI_ROOT / "DORI-Orcasound",
        DORI_ROOT / "DORI-SanctSound",
    ]

    manifest_files = DORI_ONC["filename"].apply(lambda f: Path(f).name)

    existing_files = set()
    for d in DATA_DIRS:
        repo_files = {
            p.name
            for p in d.rglob("*")
            if p.is_file() and p.suffix.lower() in (".flac", ".wav", ".mp3")
        }
        existing_files |= repo_files
        missing_mask = ~manifest_files.isin(repo_files)
        print(f"{d.name}: on_disk={len(repo_files)} missing={missing_mask.sum()} ({missing_mask.mean():.1%})")
    if VERIFY_HF:
        # check whether it is missing on hf / only on disk:
        from huggingface_hub import HfApi

        api = HfApi()
        HF_ORG = "DORI-SRKW"
        REPOS = ["DORI-ONC", "DORI-OOI", "DORI-Orcasound", "DORI-SanctSound"]

        for repo in REPOS:
            hf_files = api.list_repo_files(f"{HF_ORG}/{repo}", repo_type="dataset")
            hf_audio = {Path(f).name for f in hf_files if f.lower().endswith((".flac", ".wav", ".mp3"))}

            local_dir = DORI_ROOT / repo
            local_audio = {
                p.name for p in local_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in (".flac", ".wav", ".mp3")
            }

            missing = hf_audio - local_audio  # on HF, not downloaded -> real gap
            unexpected = local_audio - hf_audio  # on disk, not on HF -> shouldn't happen

            print(f"{repo}: on_hf={len(hf_audio)} local={len(local_audio)} "
                f"not_downloaded={len(missing)} local_only={len(unexpected)}")

    # load only local files present in DORI
    DCLDE = pd.read_csv(DCLDE_PATH)
    DORI_LOCAL = DORI_ONC[DORI_ONC['filename'].apply(lambda f: Path(f).name).isin(existing_files)]
    # We must ensure that any data overlapping with DCLDE is removed.
    # also there are some annotaitons which span the entire file, for now, we only want segment level
    # not file level annotations
    segment_rows = DORI_LOCAL[~(DORI_LOCAL['segment_start'].isna() & DORI_LOCAL['segment_end'].isna())]
    print('[local] total segment-level annotations:', len(segment_rows), '/', len(DORI_LOCAL))
    dclde_stems = set(DCLDE['Soundfile'].apply(lambda f: Path(f).stem if pd.notna(f) else None))
    non_overlap_mask = ~segment_rows['filename'].apply(lambda f: Path(f).stem).isin(dclde_stems)
    DORI_LOCAL = segment_rows[non_overlap_mask]
    print('[local] segment-level annotations not overlapping with DCLDE:', non_overlap_mask.sum(), '/', len(segment_rows))
    print(DORI_LOCAL)
    # it has been verified that the data is synced with HF, and that the associated embeddings do not account for the missing data.
    # For this reason, we will only use the DORI-ONC dataset. 
    # In particular for this version of the DORI dataset integration, we are most interested with the 
    # ONC - VENUS Node. It turns out that a portion of the DORI-ONC dataset contains data from one of:
    # {'name': 'Strait of Georgia East Node', 'lat': 49.042835, 'lon': -123.317265},
    # {'name': 'Strait of Georgia Central Node', 'lat': 49.040437, 'lon': -123.425795},
    # {'name': 'Saanich Inlet Central Node', 'lat': 48.650900, 'lon': -123.486712},

    import numpy as np

    NAMED_SITES = [
        {'name': 'Strait of Georgia East Node', 'lat': 49.042835, 'lon': -123.317265},
        {'name': 'Strait of Georgia Central Node', 'lat': 49.040437, 'lon': -123.425795},
        {'name': 'Saanich Inlet Central Node', 'lat': 48.650900, 'lon': -123.486712},
    ]
    VENUS_RADIUS_M = 1000

    def haversine_m(lat1, lon1, lat2, lon2):
        R = 6371000.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    loc_rows = DORI_LOCAL.dropna(subset=['lat', 'lon']).copy()
    loc_rows['lat_r'] = loc_rows['lat'].round(3)
    loc_rows['lon_r'] = loc_rows['lon'].round(3)

    # For every distinct (lat_r, lon_r), find the nearest named site and its distance
    _clusters_for_radius = loc_rows.groupby(['lat_r', 'lon_r']).size().reset_index(name='rows')
    _dists = pd.DataFrame({s['name']: haversine_m(_clusters_for_radius['lat_r'], _clusters_for_radius['lon_r'], s['lat'], s['lon']) for s in NAMED_SITES})
    _clusters_for_radius['min_dist'] = _dists.min(axis=1)
    _clusters_for_radius['nearest_site'] = _dists.idxmin(axis=1)

    SITE_BY_COORD = dict(
        zip(
            map(tuple, _clusters_for_radius.loc[_clusters_for_radius['min_dist'] <= VENUS_RADIUS_M, ['lat_r', 'lon_r']].values),
            _clusters_for_radius.loc[_clusters_for_radius['min_dist'] <= VENUS_RADIUS_M, 'nearest_site'],
        )
    )
    VENUS_COORDS = set(SITE_BY_COORD.keys())

    def classify_site(lat, lon):
        nearest = SITE_BY_COORD.get((round(lat, 3), round(lon, 3)))
        if nearest is None:
            return 'other hydrophone'
        return 'VENUS (Saanich Inlet)' if 'Saanich' in nearest else 'VENUS (Strait of Georgia)'

    site_clusters = (
        loc_rows.groupby(['lat_r', 'lon_r'])
        .agg(rows=('filename', 'size'), depth_mean=('depth', 'mean'))
        .reset_index()
    )
    site_clusters['site'] = site_clusters.apply(lambda r: classify_site(r['lat_r'], r['lon_r']), axis=1)

    print(site_clusters.groupby('site').agg(row_count=('rows', 'sum'), num_locations=('rows', 'count')).sort_values('row_count', ascending=False))
    print()
    site_clusters.sort_values('rows', ascending=False)
    def classify_provider(lat, lon):
        nearest = SITE_BY_COORD.get((round(lat, 3), round(lon, 3))) if pd.notna(lat) and pd.notna(lon) else None
        if nearest is None:
            return 'ONC (DORI)'
        return 'ONC VENUS (Saanich)' if 'Saanich' in nearest else 'ONC VENUS (Strait of Georgia)'

    DORI_LOCAL['provider'] = DORI_LOCAL.apply(lambda r: classify_provider(r['lat'], r['lon']), axis=1)
    print(DORI_LOCAL['provider'].value_counts())
    # some of which contain non-click human generated SRKW annotations
    # For the first version of the dataset, we will not include pseudo-labels, 
    # we will save this for experiments down the line.
    # NOTE: This is where V1 specific logic will be applied to the dataset to pull out pseudo label cases
    # and exclude click data.
    print("human generated")
    print(DORI_LOCAL[DORI_LOCAL['provider'].isin(["ONC VENUS (Strait of Georgia)", "ONC VENUS (Saanich)"])][DORI_LOCAL['call_annotation_clean'] != 'clicks'][DORI_LOCAL['ecotype_label_source'] == 'DORI (human-generated)']['ecotype_label_clean'].value_counts())
    print("pseudo-label")
    print(DORI_LOCAL[DORI_LOCAL['provider'].isin(["ONC VENUS (Strait of Georgia)", "ONC VENUS (Saanich)"])][DORI_LOCAL['call_annotation_clean'] != 'clicks'][DORI_LOCAL['ecotype_label_source'] == 'DORI (pseudo-label)']['ecotype_label_clean'].value_counts())
    # v1 contains only 3674 srkw and 1200 transient for the VENUS Deployments
    print("species level annotations for human-generated in VENUS ONC hydrophones")
    print(DORI_LOCAL[DORI_LOCAL['provider'].isin(["ONC VENUS (Strait of Georgia)", "ONC VENUS (Saanich)"])][DORI_LOCAL['call_annotation_clean'] != 'clicks'][DORI_LOCAL['species_label_source'] == 'DORI (human-generated)']['species_label_clean'].value_counts())
    print("species level annotations for human-generated in NON-VENUS ONC hydrophones")
    print(DORI_LOCAL[~DORI_LOCAL['provider'].isin(["ONC VENUS (Strait of Georgia)", "ONC VENUS (Saanich)"])][DORI_LOCAL['call_annotation_clean'] != 'clicks'][DORI_LOCAL['species_label_source'] == 'DORI (human-generated)']['species_label_clean'].value_counts())
    print("ecotype level annotations for human-generated in NON-VENUS ONC hydrophones")
    print(DORI_LOCAL[~DORI_LOCAL['provider'].isin(["ONC VENUS (Strait of Georgia)", "ONC VENUS (Saanich)"])][DORI_LOCAL['call_annotation_clean'] != 'clicks'][DORI_LOCAL['species_label_source'] == 'DORI (human-generated)']['ecotype_label_clean'].value_counts())
    # remove clicks and pseudo labels
    print(f"nrows before removing clicks and pseudo labels {len(DORI_LOCAL)}")
    import pdb;pdb.set_trace()
    DORI_LOCAL = DORI_LOCAL[DORI_LOCAL['call_annotation_clean'] != 'clicks'][DORI_LOCAL['ecotype_label_source'] == 'DORI (human-generated)']
    print(f"nrows after removing clicks and pseudo labels {len(DORI_LOCAL)}")
    import pdb;pdb.set_trace()
    def select_naive_background_windows(annotations: pd.DataFrame, window_duration: float, hop_duration: float, event_buffer: float) -> pd.DataFrame:
        """The "naive" background_method: slides a window across every annotated
        recording, keeping only starts that don't overlap a labeled event
        (+/- event_buffer). See notebooks/background_windows.ipynb."""
        valid = annotations[annotations["LocalFileOk"]].copy()

        candidates = []
        for local_path, file_rows in tqdm(valid.groupby("LocalPath"), desc="selecting background windows", unit="file"):
            file_duration = librosa.get_duration(path=local_path)
            padded = list(zip(file_rows["FileBeginSec"] - event_buffer, file_rows["FileEndSec"] + event_buffer))

            for start in np.arange(0, file_duration - window_duration, hop_duration):
                end = start + window_duration
                if any(start < pe and end > ps for ps, pe in padded):
                    continue
                candidates.append({
                    "Soundfile": file_rows["Soundfile"].iloc[0],
                    "LocalPath": local_path,
                    "Dataset": file_rows["Dataset"].iloc[0],
                    "Provider": file_rows["Provider"].iloc[0],
                    "FileBeginSec": float(start),
                    "FileEndSec": float(end),
                })
        return pd.DataFrame(candidates)

    # filename == soundfile
    DORI_LOCAL = DORI_LOCAL.rename(columns={"filename": "Soundfile"})
    # provider col in DORi is Dataset in DCLDE
    DORI_LOCAL = DORI_LOCAL.rename(columns={"provider": "Dataset"})
    DORI_LOCAL["Dataset"] = DORI_LOCAL["Dataset"].apply(lambda x: x.replace("ONC", "DORI").replace("(DORI)", ""))
    # low / high freq not provided in DORI
    DORI_LOCAL['LowFreqHz'] = pd.NA
    DORI_LOCAL['HighFreqHz'] = pd.NA
    # rename other equivelant fields.
    DORI_LOCAL = DORI_LOCAL.rename(columns={"date": "UTC", "segment_start": 'FileBeginSec', "segment_end": 'FileEndSec'})
    # match ClassSpecies logic in DCLDE dataset
    DORI_LOCAL['ecotype_label_clean'] = DORI_LOCAL['ecotype_label_clean'].fillna(DORI_LOCAL['species_label_clean'])
    # drop uncertain row
    DORI_LOCAL = DORI_LOCAL[DORI_LOCAL['ecotype_label_clean'] != "uncertain"]
    # clean up ecotype to match DCLDE
    species_rename_mapping = {
        "srkw":"SRKW",
        "humpback":"HW",
        "transient": "TKW",
        "offshore": "OKW",
        "nrkw": "NRKW",
        "orca": "KW_und",
        "noise": "AB",
        "pacific white sided dolphin": "UndBio",
        "fin whale": "UndBio",
        "sea lion": "UndBio",
        "false killer whale": "UndBio",
        "multiple classes": "UndBio",
        "sperm whale": "UndBio",
        "northern right whale dolphin": "UndBio",
        "gray": "UndBio",
        "minke": "UndBio",
        # "uncertain": "UndBio",
    }
    DORI_LOCAL['ecotype_label_clean'] = DORI_LOCAL['ecotype_label_clean'].apply(lambda x: species_rename_mapping[x] if x in species_rename_mapping else x)
    # rename to 'ClassSpecies'
    DORI_LOCAL = DORI_LOCAL.rename(columns={'ecotype_label_clean': 'ClassSpecies'})
    DORI_LOCAL['Ecotype'] = DORI_LOCAL['ClassSpecies'].where(DORI_LOCAL['ClassSpecies'].isin(DCLDE['Ecotype'].dropna().unique()), pd.NA)
    # KW cols
    DORI_LOCAL['KW_certain'] = pd.NA
    DORI_LOCAL['KW'] = DORI_LOCAL['ClassSpecies'].apply(lambda x: 1 if x in ["KW_und", "SRKW", "TKW", "OKW", "NRKW"] else 0)
    # all data currently is from ONC
    DORI_LOCAL["Provider"] = "ONC"
    # all annotations considered call level (this might be incorrect)
    DORI_LOCAL['AnnotationLevel'] = 'Call'
    # Filepath cols not relevant
    DORI_LOCAL['FilePath'] = pd.NA
    DORI_LOCAL['FileOk'] = True
    # calltype
    DORI_LOCAL=DORI_LOCAL.rename(columns={'call_annotation_clean': 'CallType'})
    # not adding calltype category for now.
    DORI_LOCAL['CalltypeCategory'] = pd.NA
    # just make everything false as there are no questions in the DS
    DORI_LOCAL['HasQ'] = False
    DORI_LOCAL['CalltypeHasQ'] = False
    # localpath / local files
    FILE_ROOT = DORI_ROOT / 'DORI-ONC'
    import os
    stem = lambda f: os.path.splitext(os.path.basename(f))[0]
    paths = {stem(f): os.path.join(d, f) for d, _, fs in os.walk(FILE_ROOT) for f in fs if f.endswith('.flac')}
    DORI_LOCAL['LocalPath'] = DORI_LOCAL['Soundfile'].map(lambda f: paths.get(stem(f)))
    DORI_LOCAL['LocalFileOk'] = DORI_LOCAL['LocalPath'].notna()
    # centertime and duration:
    DORI_LOCAL['CenterTime'] = (DORI_LOCAL['FileEndSec'] + DORI_LOCAL['FileBeginSec']) / 2
    DORI_LOCAL['Duration'] = DORI_LOCAL['FileEndSec'] - DORI_LOCAL['FileBeginSec']
    DORI_LOCAL['EcotypeCertain'] = pd.NA
    # labels, same idea as derive_labels in the DCLDE build script
    DORI_LOCAL['Labels'] = DORI_LOCAL['ClassSpecies']
    DORI_LOCAL['BackgroundMethod'] = pd.NA
    # background windows: "naive" method, using select_naive_background_windows defined at the top of this cell
    import librosa
    import numpy as np
    from tqdm import tqdm
    # file-level annotations (no segment times) cover the whole file, so those files give no background
    # use full DORI_ONC to ensure we are excluding all annotations including those which
    #  are discarded in the previous steps of the process
    events = DORI_ONC.rename(columns={"filename": "Soundfile", "segment_start": "FileBeginSec", "segment_end": "FileEndSec"})
    # background is only drawn from the files kept in DORI_LOCAL, which also carry the path / Dataset / Provider cols
    events = events.merge(DORI_LOCAL[["Soundfile", "LocalPath", "LocalFileOk", "Dataset", "Provider"]].drop_duplicates("Soundfile"), on="Soundfile")
    events = events.fillna({'FileBeginSec': 0, 'FileEndSec': float('inf')})
    if BACKGROUND_METHOD != "naive":
        raise NotImplementedError(f"BACKGROUND_METHOD={BACKGROUND_METHOD!r} not implemented yet -- only 'naive' exists so far.")
    background = select_naive_background_windows(
        events, window_duration=BACKGROUND_WINDOW_DURATION, hop_duration=BACKGROUND_HOP_DURATION, event_buffer=BACKGROUND_EVENT_BUFFER,
    )
    # fill in the remaining cols the same way to_annotations_schema does in the DCLDE build script
    background['Duration'] = background['FileEndSec'] - background['FileBeginSec']
    background['CenterTime'] = (background['FileBeginSec'] + background['FileEndSec']) / 2
    background['Labels'] = BACKGROUND_LABEL
    background['BackgroundMethod'] = BACKGROUND_METHOD
    background['KW'] = 0
    background[['FileOk', 'LocalFileOk']] = True
    background[['HasQ', 'CalltypeHasQ', 'EcotypeCertain']] = False
    background
    DORI_LOCAL = pd.concat([DORI_LOCAL, background], ignore_index=True)
    print(DORI_LOCAL['Labels'].value_counts(dropna=False))
    DORI_LOCAL['uid'] = DORI_LOCAL.index
    set(DCLDE.columns) - set(DORI_LOCAL.columns)
    # merge DCLDE and DORI_ONC; cols that only exist in DORI_ONC are NaN on the DCLDE rows
    COMBINED = pd.concat([DCLDE, DORI_LOCAL], ignore_index=True)
    # uid was the row index in each source, so the two overlap; reassign
    COMBINED['uid'] = COMBINED.index

    # save + log to MLflow's shared 'Datasets' experiment, as build_dclde_2027_annotations.py does
    run_dir.mkdir(parents=True)
    configure_datasets_mlflow(MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name=run_dir.name):
        mlflow.log_params({
            "dclde_path": DCLDE_PATH,
            "dori_root": str(DORI_ROOT),
            "run_dir": str(run_dir),
            "verify_hf": VERIFY_HF,
            "background_method": BACKGROUND_METHOD,
            "background_window_duration": BACKGROUND_WINDOW_DURATION,
            "background_hop_duration": BACKGROUND_HOP_DURATION,
            "background_event_buffer": BACKGROUND_EVENT_BUFFER,
            "git_commit": git_commit(SCRIPT_DIR),
            "command": f"{sys.executable} {Path(__file__).resolve()}",
        })

        out_path = run_dir / "annotations.csv"
        COMBINED.to_csv(out_path, index=False)
        mlflow.log_param("annotations_path", str(out_path))
        mlflow.log_metric("n_rows", len(COMBINED))
        mlflow.log_metric("n_background_rows", int((COMBINED["Labels"] == BACKGROUND_LABEL).sum()))
        print(f"Saved {len(COMBINED):,} rows, {len(COMBINED.columns)} columns -> {out_path}")
