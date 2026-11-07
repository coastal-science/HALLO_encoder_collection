# DORI Dataset
The integration script for this dataset is focussed on gathering the ONC VENUS data.

The dataset builds on top of the DCLDE dataset, so the annotation.csv file must be passed to the build_DORI_annotations.py script. 

checkout experiment/eda.ipynb for all info on how the build script was created.

## Downloading missing DORI audio: `download_dori_missing.py`
Brings `/data/DORI` up to date with Bret's newer `DORI.csv` (`/home/noah/ONC_download_project/DORI.csv`, not on HF yet).
The gaps were found in `/home/noah/ONC_download_project/compare_dori_csv.ipynb`.

| step | what | size (2026-11-06 plan) |
|---|---|---|
| `hf` | files on the `DORI-SRKW` HF repos missing locally: 491 DORI-OOI, 10 DORI-ONC | 12.4 GB |
| `noaa` | 32,498 SanctSound / ONMS / NRS 5 min clips, cut from 1,044 parent recordings in `gs://noaa-passive-bioacoustic` -> `/data/DORI/DORI-SanctSound/data` | 0.44 TB download, roughly 200 GB kept |
| left over | 121 `unavailable` (ONC / OOI files on neither HF nor NOAA: need the ONC API) and 1 `no_parent` clip | - |

1. Keep `LIST_ONLY = True` for the first run. It writes `/data/DORI/logs/download_dori_missing/plan.csv` and nothing else.
2. Set `LIST_ONLY = False`, then run `nohup .venv/bin/python scripts/DORI/download_dori_missing.py &`. Progress goes to `log.txt` and `progress.csv` in the same log folder. The script can be restarted, because finished files are skipped.
3. Check the 456 `off_grid` clips, which don't start on the parent's 5 min grid. The whisper-tiny embeddings in `/data/DORI/DORI-SanctSound/whisper-tiny_embeddings_SanctSound.parquet` cover these files, so re-embedding a sample is one way to confirm the cuts match DORI's.

Set `INCLUDE_SNAPSHOT = True` to also cut the snapshot's NOAA clips (about 0.47 TB more).
