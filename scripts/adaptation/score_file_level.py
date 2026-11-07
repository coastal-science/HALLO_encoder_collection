"""File-level scoring of the deployed hallo-tuned-detectors models on the leak-free holdout
(make_file_level_holdout.py), keeping every window / frame score so thresholds can be swept afterwards.

The deployed decision is reproduced with the backends' own code and configs (hallo-tuned-detectors/configs):
  birdnet   predict_file over the whole file (3 s windows, hop 1.5 s, channel 0) with every class threshold at 0, so all
            window scores are kept; then the deployed thresholds (SRKW 0.95, TKW 0.9, HW 0.9) are applied and the
            backend's own summarize() makes the file call (majority vote of accepted windows).
  sequence  the backend's compute_spectrogram + process_with_seq (whole minutes); file = KW iff the number of
            find_peaks(height=0.75) peaks over all minutes >= 10 (min_peaks), exactly as backends.sequence.detect.
--verify N re-runs the unmodified backend detect() on the first N files and checks the file calls match.

Run in hallo-tuned-detectors/.venv (TF 2.8 / ketos). Resumable: one .npz per file under <out>/<detector>/windows/;
--shard i --n-shards n splits the file list for parallel CPU workers. --merge writes <out>/<detector>_files.csv.

    PY=~/hallo-tuned-detectors/.venv/bin/python
    $PY scripts/adaptation/score_file_level.py --detector birdnet --shard 0 --n-shards 8
    $PY scripts/adaptation/score_file_level.py --detector birdnet --merge
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "scripts/adaptation/results/file_level"
HOLDOUT = OUT / "sog_holdout_24h.csv"
DETECTORS_REPO = Path.home() / "hallo-tuned-detectors"
BIRDNET = dict(model="models/kait_birdnet_class/v1/birdnet07_8khz_cutoff.tflite", confidence_thresh=0.6,
               class_thresholds={"HW": 0.9, "SRKW": 0.95, "TKW": 0.9}, hop_s=1.5, channels=0, batch_size=16)
SEQUENCE = dict(model="models/hallo_seq_kw_det/v1/hallo_seq_kw_det_v1.kt", peak_height=0.75, min_peaks=10)


class BirdNet:
    def __init__(self):
        from backends.birdnet import detector
        self.d = detector
        self.m = detector.BirdNetDetector(DETECTORS_REPO / BIRDNET["model"])
        self.m.confidence_thresh, self.m.class_thresholds = 0.0, {lab: 0.0 for lab in self.m.labels}  # keep every window

    def score(self, path: str) -> dict:
        run = self.m.predict_file(path, hop_s=BIRDNET["hop_s"], channels=BIRDNET["channels"], batch_size=BIRDNET["batch_size"])
        p = run.predictions
        return dict(begin_s=p["Begin Time (S)"].to_numpy(float), scores=p[self.m.labels].to_numpy(np.float32),
                    labels=np.array(self.m.labels), n_windows=run.n_windows)

    @staticmethod
    def decide(path: str, w: dict) -> dict:
        """The deployed file call from saved window scores (backend summarize() on the thresholded windows)."""
        from backends.birdnet import detector
        labels = [str(x) for x in w["labels"]]
        s = w["scores"]
        rows = pd.DataFrame(s, columns=labels)
        if len(rows):
            top = s.argmax(1)
            rows["Predicted Class"] = np.array(labels)[top]
            rows["Top Score"] = s.max(1)
            thresh = np.array([BIRDNET["class_thresholds"].get(lab, BIRDNET["confidence_thresh"]) for lab in labels])[top]
            rows = rows[rows["Top Score"] >= thresh]
        summary = detector.BirdNetDetector.summarize(None, path, detector.PredictionRun(rows, int(w["n_windows"])))
        return dict(birdnet_species=summary.species, birdnet_ecotype=summary.ecotype, birdnet_n_accepted=summary.n_detections,
                    birdnet_n_windows=summary.n_windows)


class Sequence:
    def __init__(self):
        from backends.sequence import detector
        self.d = detector
        self.m = detector.load_model(DETECTORS_REPO / SEQUENCE["model"])

    def score(self, path: str) -> dict:
        minutes = self.d.compute_spectrogram(path, self.d.DEFAULT_SPEC_PARAMETERS)
        frames = [np.atleast_1d(s).astype(np.float32) for s in self.d.process_with_seq(self.m, minutes)]
        return dict(frames=np.stack(frames) if frames else np.zeros((0, 0), np.float32), n_minutes=len(frames))

    @staticmethod
    def decide(path: str, w: dict) -> dict:
        from scipy.signal import find_peaks
        n = sum(len(find_peaks(m, height=SEQUENCE["peak_height"])[0]) for m in w["frames"])
        return dict(sequence_species="KW" if n >= SEQUENCE["min_peaks"] else "", sequence_n_peaks=n,
                    sequence_n_minutes=int(w["n_minutes"]))


DETECTORS = {"birdnet": BirdNet, "sequence": Sequence}


def verify(name: str, files: pd.DataFrame, out: Path) -> None:
    """Unmodified backend detect() vs decide() from the saved scores."""
    model = DETECTORS[name]()
    backend = model.d
    params = ({k: BIRDNET[k] for k in ("confidence_thresh", "class_thresholds", "hop_s", "channels", "batch_size")}
              if name == "birdnet" else dict(peak_height=SEQUENCE["peak_height"], min_peaks=SEQUENCE["min_peaks"]))
    bad = 0
    for path in files.LocalPath:
        with tempfile.TemporaryDirectory() as d:
            if name == "birdnet":
                ref_model = backend.load_model(DETECTORS_REPO / BIRDNET["model"])
                s = backend.detect(path, output_dir=d, model=ref_model, **params)
                ref = (s.species, s.ecotype)
            else:
                s = backend.detect(path, output_dir=d, model=model.m, **params)
                ref = (s.species, s.ecotype)
        w = dict(np.load(out / "windows" / (Path(path).name + ".npz"), allow_pickle=False))
        got = DETECTORS[name].decide(path, w)
        mine = (got[f"{name}_species"], got.get(f"{name}_ecotype", ""))
        ok = mine == ref
        bad += not ok
        print(f"  verify {'OK ' if ok else 'MISMATCH'} {Path(path).name}: backend {ref} / ours {mine}", flush=True)
    print(f"verify: {len(files) - bad}/{len(files)} match")
    if bad:
        sys.exit(1)


def merge(name: str, files: pd.DataFrame, out: Path) -> None:
    rows, missing = [], []
    for r in files.itertuples():
        f = out / "windows" / (r.filename + ".npz")
        if not f.exists():
            missing.append(r.filename)
            continue
        rows.append(dict(filename=r.filename, truth=r.truth, **DETECTORS[name].decide(r.LocalPath, dict(np.load(f)))))
    df = pd.DataFrame(rows)
    path = OUT / f"{name}_files.csv"
    df.to_csv(path, index=False)
    print(f"wrote {len(df)} files to {path}; {len(missing)} not scored yet")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--detector", choices=DETECTORS, required=True)
    p.add_argument("--holdout", type=Path, default=HOLDOUT)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--limit-files", type=int, help="timing / smoke test: first N files of this shard")
    p.add_argument("--verify", type=int, metavar="N", help="check the first N scored files against the backend's detect()")
    p.add_argument("--merge", action="store_true")
    args = p.parse_args()
    sys.path.insert(0, str(DETECTORS_REPO / "src"))

    files = pd.read_csv(args.holdout)
    out = OUT / args.detector
    (out / "windows").mkdir(parents=True, exist_ok=True)
    if args.merge:
        return merge(args.detector, files, out)
    if args.verify:
        scored = files[files.filename.map(lambda n: (out / "windows" / (n + ".npz")).exists())]
        return verify(args.detector, scored.head(args.verify), out)

    mine = files.iloc[args.shard::args.n_shards]
    if args.limit_files:
        mine = mine.head(args.limit_files)
    todo = [r for r in mine.itertuples() if not (out / "windows" / (r.filename + ".npz")).exists()]
    print(f"{args.detector} shard {args.shard}/{args.n_shards}: {len(mine)} files, {len(todo)} to do", flush=True)
    model = DETECTORS[args.detector]()
    t0, failed = time.time(), []
    for i, r in enumerate(todo, 1):
        try:
            w = model.score(r.LocalPath)
        except Exception as e:
            failed.append(f"{r.LocalPath}: {type(e).__name__}: {e}")
            print("  FAILED", failed[-1], flush=True)
            continue
        tmp = out / "windows" / (r.filename + ".tmp.npz")
        np.savez_compressed(tmp, **w)
        tmp.rename(out / "windows" / (r.filename + ".npz"))
        if i % 10 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  {i}/{len(todo)} files, {el:.0f} s ({el / i:.1f} s/file)", flush=True)
    print(f"done: {len(todo) - len(failed)} scored, {len(failed)} failed, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
