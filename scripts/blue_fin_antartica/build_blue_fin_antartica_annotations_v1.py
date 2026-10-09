# HALLO Encoder Collection IWC-SORP Antarctic blue / fin whale library integration v1 script

import re
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from encoder_pipeline.common.mlflow_utils import configure_datasets_mlflow

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

DATASET_NAME = "blue_fin_antartica"
ROOT = Path("/data/blue_fin_antartica")
DATA_DIR = REPO_ROOT / "data_raw" / DATASET_NAME
MLFLOW_TRACKING_URI = "http://localhost:8989"
MLFLOW_EXPERIMENT_NAME = f"Datasets/{DATASET_NAME}"
PROVIDER = "IWC_SORP"
# How to select 'Background' windows; only 'naive' is implemented so far.
BACKGROUND_METHOD = "naive"
BACKGROUND_WINDOW_DURATION = 5.0
BACKGROUND_HOP_DURATION = 2.5
BACKGROUND_EVENT_BUFFER = 10.0
BACKGROUND_LABEL = "Background"
# suffix in the Greenwich64S2015 selection tables' file names which the wav files don't have
GREENWICH_SUFFIX = "_AWI229-11_SV1057"

# (pattern on the lowercased selection table filename, species, call type), first match wins
SELECTION_TABLE_RULES = [
    (r"ant-?a", "BlueWhale", "Ant-A"),
    (r"ant-?b", "BlueWhale", "Ant-B"),
    (r"ant-?z", "BlueWhale", "Ant-Z"),
    (r"bm[._]?d", "BlueWhale", "D"),
    (r"bm_swi", "BlueWhale", "SWI"),
    (r"20plus", "FinWhale", "20Plus"),
    (r"highercall", "FinWhale", "HigherCall"),
    (r"(bp|fin)[._-]?20", "FinWhale", "20Hz"),
    (r"(bp|fin)[._-]?(downsweep|ds|dswp|dwnswp)", "FinWhale", "Downsweep"),
    (r"backbeat", "FinWhale", "Backbeat"),
    (r"hump", "HW", None),
    (r"bioduckanddownsweeps", "MinkeWhale", "BioduckAndDownsweeps"),
    (r"minkebio", "MinkeWhale", "Bioduck"),
    (r"minkeds", "MinkeWhale", "Downsweep"),
    (r"minke", "MinkeWhale", None),
    (r"tonal30hz", "UndBio", "Tonal30Hz"),
    (r"recurrent", "UndBio", "RecurrentPulses"),
    (r"swi", "UndBio", "SWI"),
    (r"unid|unknown", "UndBio", None),
]

ANNOTATION_COLUMNS = [
    "Soundfile", "Dataset", "LowFreqHz", "HighFreqHz", "FileEndSec", "UTC", "FileBeginSec", "ClassSpecies",
    "KW", "KW_certain", "Ecotype", "Provider", "AnnotationLevel", "FilePath", "FileOk", "CallType",
    "CalltypeCategory", "HasQ", "CalltypeHasQ", "LocalPath", "LocalFileOk", "CenterTime", "Duration",
    "EcotypeCertain", "Labels", "BackgroundMethod", "uid",
]


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
    check_uncommitted_changes(SCRIPT_DIR)
    run_dir = DATA_DIR / time.strftime("%Y%m%d_%H%M%S")

    # each site dir holds one Raven selection table per call type; the species comes from the table's filename.
    # empty tables are skipped, and only the first table is used when a site has duplicates.
    tables = []
    for site_dir in sorted(d for d in ROOT.iterdir() if (d / "wav").is_dir()):
        seen = set()
        for path in sorted(site_dir.glob("*.txt")):
            species, call_type = next((s, c) for pattern, s, c in SELECTION_TABLE_RULES if re.search(pattern, path.name.lower()))
            table = pd.read_csv(path, sep="\t")
            if table.empty or (species, call_type) in seen:
                continue
            seen.add((species, call_type))
            table["Dataset"] = site_dir.name
            table["ClassSpecies"] = species
            table["CallType"] = call_type
            tables.append(table)
    BLUE_FIN = pd.concat(tables, ignore_index=True)
    for col in ["Begin File", "End File"]:
        BLUE_FIN[col] = BLUE_FIN[col].str.replace(GREENWICH_SUFFIX, "", regex=False)

    # localpath / local files
    wav_paths = {(p.parent.parent.name, p.name.lower()): str(p) for p in ROOT.glob("*/wav/*") if p.suffix.lower() == ".wav"}
    BLUE_FIN["Soundfile"] = BLUE_FIN["Begin File"]
    BLUE_FIN["LocalPath"] = [wav_paths.get((d, f.lower())) for d, f in zip(BLUE_FIN["Dataset"], BLUE_FIN["Soundfile"])]
    BLUE_FIN["LocalFileOk"] = BLUE_FIN["LocalPath"].notna()

    # 'Begin Time (s)' is cumulative over the whole site, so the in-file times come from the sample cols
    sample_rate = BLUE_FIN["LocalPath"].map({p: sf.info(p).samplerate for p in BLUE_FIN["LocalPath"].dropna().unique()})
    BLUE_FIN["FileBeginSec"] = BLUE_FIN["Beg File Samp (samples)"] / sample_rate
    BLUE_FIN["FileEndSec"] = BLUE_FIN["End File Samp (samples)"] / sample_rate

    # annotations spanning two files are dropped, but background must still avoid both of their halves
    spans_files = BLUE_FIN["Begin File"] != BLUE_FIN["End File"]
    heads = BLUE_FIN[spans_files].assign(FileEndSec=float("inf"))
    tails = BLUE_FIN[spans_files].assign(Soundfile=BLUE_FIN["End File"], FileBeginSec=0.0)
    tails["LocalPath"] = [wav_paths.get((d, f.lower())) for d, f in zip(tails["Dataset"], tails["Soundfile"])]
    print(f"nrows before dropping annotations spanning two files {len(BLUE_FIN)}")
    BLUE_FIN = BLUE_FIN[~spans_files]
    BLUE_FIN = BLUE_FIN[BLUE_FIN["FileEndSec"] > BLUE_FIN["FileBeginSec"]]
    print(f"nrows after dropping annotations spanning two files / with no duration {len(BLUE_FIN)}")
    events = pd.concat([BLUE_FIN, heads, tails], ignore_index=True)
    events["Provider"] = PROVIDER

    BLUE_FIN = BLUE_FIN.rename(columns={"Low Freq (Hz)": "LowFreqHz", "High Freq (Hz)": "HighFreqHz"})
    BLUE_FIN["UTC"] = pd.to_datetime(BLUE_FIN["Begin Date Time"].str.split().str.join(" "), format="mixed")
    BLUE_FIN["Provider"] = PROVIDER
    BLUE_FIN["AnnotationLevel"] = "Call"
    BLUE_FIN["Labels"] = BLUE_FIN["ClassSpecies"]
    BLUE_FIN["CenterTime"] = (BLUE_FIN["FileEndSec"] + BLUE_FIN["FileBeginSec"]) / 2
    BLUE_FIN["Duration"] = BLUE_FIN["FileEndSec"] - BLUE_FIN["FileBeginSec"]
    BLUE_FIN["KW"] = 0
    BLUE_FIN["FileOk"] = True
    BLUE_FIN[["HasQ", "CalltypeHasQ", "EcotypeCertain"]] = False
    BLUE_FIN[["KW_certain", "Ecotype", "FilePath", "CalltypeCategory", "BackgroundMethod"]] = pd.NA

    def select_naive_background_windows(annotations: pd.DataFrame, window_duration: float, hop_duration: float, event_buffer: float) -> pd.DataFrame:
        """The "naive" background_method: slides a window across every annotated
        recording, keeping only starts that don't overlap a labeled event
        (+/- event_buffer). See notebooks/background_windows.ipynb."""
        valid = annotations[annotations["LocalFileOk"]].copy()

        candidates = []
        for local_path, file_rows in tqdm(valid.groupby("LocalPath"), desc="selecting background windows", unit="file"):
            file_duration = sf.info(local_path).duration
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

    if BACKGROUND_METHOD != "naive":
        raise NotImplementedError(f"BACKGROUND_METHOD={BACKGROUND_METHOD!r} not implemented yet -- only 'naive' exists so far.")
    background = select_naive_background_windows(
        events, window_duration=BACKGROUND_WINDOW_DURATION, hop_duration=BACKGROUND_HOP_DURATION, event_buffer=BACKGROUND_EVENT_BUFFER,
    )
    # fill in the remaining cols the same way to_annotations_schema does in the DCLDE build script
    background["Duration"] = background["FileEndSec"] - background["FileBeginSec"]
    background["CenterTime"] = (background["FileBeginSec"] + background["FileEndSec"]) / 2
    background["Labels"] = BACKGROUND_LABEL
    background["BackgroundMethod"] = BACKGROUND_METHOD
    background["KW"] = 0
    background[["FileOk", "LocalFileOk"]] = True
    background[["HasQ", "CalltypeHasQ", "EcotypeCertain"]] = False

    BLUE_FIN = pd.concat([BLUE_FIN, background], ignore_index=True)
    BLUE_FIN["uid"] = BLUE_FIN.index
    BLUE_FIN = BLUE_FIN[ANNOTATION_COLUMNS]
    print(BLUE_FIN.groupby("Dataset")["Labels"].value_counts().unstack(fill_value=0))

    # save + log to MLflow's shared 'Datasets' experiment, as build_dclde_2027_annotations.py does
    run_dir.mkdir(parents=True)
    configure_datasets_mlflow(MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name=run_dir.name):
        mlflow.log_params({
            "root": str(ROOT),
            "run_dir": str(run_dir),
            "background_method": BACKGROUND_METHOD,
            "background_window_duration": BACKGROUND_WINDOW_DURATION,
            "background_hop_duration": BACKGROUND_HOP_DURATION,
            "background_event_buffer": BACKGROUND_EVENT_BUFFER,
            "git_commit": git_commit(SCRIPT_DIR),
            "command": f"{sys.executable} {Path(__file__).resolve()}",
        })

        out_path = run_dir / "annotations.csv"
        BLUE_FIN.to_csv(out_path, index=False)
        mlflow.log_param("annotations_path", str(out_path))
        mlflow.log_metric("n_rows", len(BLUE_FIN))
        mlflow.log_metric("n_background_rows", int((BLUE_FIN["Labels"] == BACKGROUND_LABEL).sum()))
        print(f"Saved {len(BLUE_FIN):,} rows, {len(BLUE_FIN.columns)} columns -> {out_path}")
