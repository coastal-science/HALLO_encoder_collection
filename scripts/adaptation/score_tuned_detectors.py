"""Score the off-the-shelf hallo-tuned-detectors models on the pretraining comparison's clips, so they can be compared
with the fine-tuned arms (compare_tuned_detectors.py) on exactly the same val / test rows.

One detector per invocation, because the backends need different environments:
  hallo_v1   models/hallo_encoder/v1 (EffNet-B1, dclde-supervised-1)    -> embed_venv (torch + encoder_pipeline), GPU
  birdnet    models/kait_birdnet_class/v1 (BirdNET 07 tflite)            -> hallo-tuned-detectors/.venv (TF 2.8), CPU
  sequence   models/hallo_seq_kw_det/v1 (1D CNN + GRU, KW yes / no)       -> hallo-tuned-detectors/.venv (ketos), CPU

Clips = every uid that is val or test in any fold of splits_pretrain_cmp_ft_<site>.csv. The models are not trained on the
site, so each clip is scored once and compare_tuned_detectors.py copies the score to every fold that holds the clip.
Each clip is windowed as the arms' preprocessor does (encoder_pipeline.preprocessor.annotation.Annotation: centred on
FileBeginSec + Duration / 2, clamped inside the file), with each model's own input length and sample rate:
  hallo_v1   5 s at 32 kHz (its training config: time_offset 2.5, resample_sr 32000). NB the hallo_encoder backend in
             hallo-tuned-detectors loads audio at the file's native rate, which changes the STFT hop / mel scale in
             seconds on 64 / 128 kHz files; here we resample as in training.
  birdnet    3 s at 48 kHz, sigmoid scores per class (the backend's predict_file).
  sequence   the backend's own whole-file spectrogram (compute_spectrogram: 400 Hz high-pass, per-file normalisation,
             one 512 x 128 input per minute); a clip's score = max frame score inside the 3 s arm window around it.

Writes <out-dir>/<detector>_<site>.csv: uid, LocalPath, center_s, and the model's scores (prob_* columns), and logs it as
an MLflow run "score_<detector>_<site>" in experiment hallo_tuned_detectors_comparison (params: model path / version.json,
window, sample rate; metrics: clip / failure counts; artifacts: the csv + the model's version.json). --no-mlflow skips it.

    python scripts/adaptation/score_tuned_detectors.py --detector hallo_v1 \
        --detectors-repo ~/hallo-tuned-detectors --path-map /data/DORI=/home/noah/datasets/DORI [--limit-files 5]
"""
import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "data_raw/DORI/20261022_190549"
SITES = {"strait_of_georgia": "dori_venus_strait_of_georgia", "limekiln": "limekiln", "tekteksen": "tekteksen"}
ARM_WINDOW_S = 3.0  # the SSAMBA arms' window (time_offset 1.5), used for the sequence model's per-clip max
BIRDNET_THRESHOLDS = dict(confidence_thresh=0.6, class_thresholds={"HW": 0.9, "SRKW": 0.95, "TKW": 0.9})  # configs/birdnet/base.json


def clips(site: str) -> pd.DataFrame:
    dep = SITES[site]
    splits = pd.read_csv(RUN / f"splits_pretrain_cmp_ft_{dep}.csv")
    keep = splits[splits.filter(like="fold_").isin(["val", "test"]).any(axis=1)].uid
    ann = pd.read_csv(RUN / f"annotations_bgcap10000_ft_{dep}.csv", usecols=["uid", "LocalPath", "FileBeginSec", "Duration"])
    ann = ann[ann.uid.isin(keep)].copy()
    ann["center_s"] = ann.FileBeginSec + ann.Duration / 2
    return ann[["uid", "LocalPath", "center_s"]].sort_values(["LocalPath", "center_s"]).reset_index(drop=True)


def window(signal: np.ndarray, sr: int, center: float, length_s: float) -> np.ndarray:
    """Annotation's clamp: start = max(0, min(center - L/2, file_dur - L)); zero-padded if the file is shorter."""
    n = int(length_s * sr)
    start = int(max(0.0, min(center - length_s / 2, len(signal) / sr - length_s)) * sr)
    seg = signal[start:start + n]
    return np.pad(seg, (0, n - len(seg))) if len(seg) < n else seg


EXPERIMENT = "hallo_tuned_detectors_comparison"
TRACKING_URI = "http://localhost:5000"


class HalloV1:
    rate, length_s, labels = 32000, 5.0, ["Background", "HW", "SRKW", "TKW"]
    model_rel = "models/hallo_encoder/v1/fold0_best.pt"

    def __init__(self, repo: Path):
        import torch
        from types import SimpleNamespace
        sys.path.insert(0, str(repo / "src"))
        from backends.hallo_encoder.detector import HALLOEncoderModel
        self.torch, self.ns = torch, SimpleNamespace
        self.m = HALLOEncoderModel(repo / self.model_rel)

    def score(self, path: str, centers: np.ndarray) -> np.ndarray:
        signal, _ = librosa.load(path, sr=self.rate)
        specs = np.stack([self.m._compute_spectrogram(window(signal, self.rate, c, self.length_s), self.rate) for c in centers])
        with self.torch.no_grad():
            logits = self.m.model(self.torch.from_numpy(specs).float().unsqueeze(1).to(self.m.device))
        return self.torch.softmax(logits, dim=1).cpu().numpy()


class BirdNet:
    rate, labels = 48000, ["HW", "SRKW", "TKW"]
    model_rel = "models/kait_birdnet_class/v1/birdnet07_8khz_cutoff.tflite"

    def __init__(self, repo: Path):
        import scipy.special
        sys.path.insert(0, str(repo / "src"))
        from backends.birdnet.detector import CLASS_LABELS, BirdNetDetector
        assert [CLASS_LABELS[i] for i in sorted(CLASS_LABELS)] == self.labels
        self.expit = scipy.special.expit
        self.m = BirdNetDetector(repo / self.model_rel, **BIRDNET_THRESHOLDS)
        self.length_s = self.m.audio_duration

    def score(self, path: str, centers: np.ndarray) -> np.ndarray:
        signal, _ = librosa.load(path, sr=self.rate)
        batch = np.stack([window(signal, self.rate, c, self.m.audio_duration) for c in centers]).astype(np.float32)
        return np.concatenate([self.expit(self.m._predict_batch(batch[i:i + 64])) for i in range(0, len(batch), 64)])


class Sequence:
    labels, rate, length_s = ["KW"], 48000, 60.0  # whole-minute inputs; per-clip score = max over the 3 s arm window
    model_rel = "models/hallo_seq_kw_det/v1/hallo_seq_kw_det_v1.kt"

    def __init__(self, repo: Path):
        sys.path.insert(0, str(repo / "src"))
        from backends.sequence import detector
        self.d = detector
        self.m = detector.load_model(repo / self.model_rel)

    def score(self, path: str, centers: np.ndarray) -> np.ndarray:
        minutes = self.d.compute_spectrogram(path, self.d.DEFAULT_SPEC_PARAMETERS)  # whole minutes only, as the backend
        frames = np.concatenate([np.atleast_1d(s) for s in self.d.process_with_seq(self.m, minutes)])
        t = (np.arange(len(frames)) + 0.5) * (60.0 * len(minutes) / len(frames))  # frame centre times, s
        out = np.full((len(centers), 1), np.nan)
        for i, c in enumerate(centers):
            lo = max(0.0, min(c - ARM_WINDOW_S / 2, t[-1] + 1e-6 - ARM_WINDOW_S))
            inside = (t >= lo) & (t <= lo + ARM_WINDOW_S)
            if inside.any():  # clips in a trailing partial minute are not scored by the backend -> NaN
                out[i, 0] = frames[inside].max()
        return out


DETECTORS = {"hallo_v1": HalloV1, "birdnet": BirdNet, "sequence": Sequence}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--detector", choices=DETECTORS, required=True)
    p.add_argument("--site", choices=SITES, default="strait_of_georgia")
    p.add_argument("--detectors-repo", type=Path, default=Path.home() / "hallo-tuned-detectors")
    p.add_argument("--path-map", action="append", default=[], help="OLD=NEW prefix rewrite of LocalPath (repeatable)")
    p.add_argument("--out-dir", type=Path, default=REPO / "scripts/adaptation/results/tuned_detectors")
    p.add_argument("--limit-files", type=int, help="smoke test: only the first N files")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--tracking-uri", default=TRACKING_URI)
    p.add_argument("--experiment", default=EXPERIMENT)
    args = p.parse_args()

    table = clips(args.site)
    for rule in args.path_map:
        old, new = rule.split("=", 1)
        table["LocalPath"] = table.LocalPath.str.replace(old, new, n=1, regex=False)
    files = list(table.groupby("LocalPath", sort=False))
    if args.limit_files:
        files = files[:args.limit_files]
    model = DETECTORS[args.detector](args.detectors_repo.expanduser())
    out = args.out_dir / f"{args.detector}_{args.site}{'_smoke' if args.limit_files else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{args.detector} / {args.site}: {sum(len(g) for _, g in files)} clips in {len(files)} files -> {out}", flush=True)

    frames, failed, t0 = [], [], time.time()
    for i, (path, group) in enumerate(files, 1):
        try:
            scores = model.score(path, group.center_s.to_numpy())
        except Exception as e:
            failed.append(f"{path}: {type(e).__name__}: {e}")
            continue
        frames.append(group.assign(**{f"prob_{l}": scores[:, j] for j, l in enumerate(model.labels)}))
        if i % 100 == 0 or i == len(files):
            print(f"  {i}/{len(files)} files, {time.time() - t0:.0f} s", flush=True)
    for f in failed[:20]:
        print("  FAILED", f)
    if not frames:
        sys.exit(f"all {len(failed)} files failed")
    result = pd.concat(frames)
    result.to_csv(out, index=False)
    nan = int(result.filter(like="prob_").isna().any(axis=1).sum())
    print(f"wrote {len(result)} clips ({nan} unscored) to {out}; {len(failed)} files failed")
    if not args.no_mlflow:
        log_run(args, model, out, failed, dict(n_files=len(files), n_clips=len(result), n_unscored=nan,
                                               n_failed_files=len(failed), runtime_s=time.time() - t0))


def log_run(args, model, out: Path, failed: list[str], metrics: dict) -> None:
    import json
    import subprocess
    import tempfile

    import mlflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    repo = args.detectors_repo.expanduser()
    model_path = repo / model.model_rel
    version = model_path.parent / "version.json"
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    name = f"score_{args.detector}_{args.site}" + ("_smoke" if args.limit_files else "")
    with mlflow.start_run(run_name=name):
        mlflow.set_tags(dict(stage="score", detector=args.detector, site=args.site, smoke=str(bool(args.limit_files))))
        params = dict(detector=args.detector, site=args.site, model_path=str(model_path), detectors_repo_commit=commit,
                      sample_rate=model.rate, window_s=model.length_s, classes=",".join(model.labels),
                      path_map=";".join(args.path_map), limit_files=args.limit_files)
        if args.detector == "birdnet":
            params["birdnet_thresholds"] = json.dumps(BIRDNET_THRESHOLDS)
        if version.exists():
            params.update({f"version.{k}": v for k, v in json.loads(version.read_text()).items() if not isinstance(v, (dict, list))})
            mlflow.log_artifact(str(version))
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(out))
        if failed:
            with tempfile.TemporaryDirectory() as d:
                (Path(d) / "failed_files.txt").write_text("\n".join(failed))
                mlflow.log_artifact(str(Path(d) / "failed_files.txt"))
    print(f"logged MLflow run {name} to experiment {args.experiment}")


if __name__ == "__main__":
    main()
