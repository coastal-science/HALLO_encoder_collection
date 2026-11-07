"""Convert Google's Perch v2 TF SavedModel into PyTorch weights for encoder_pipeline.model_trainer.perch_v2, checked
against the SavedModel itself before anything is written.

The SavedModel (kagglehub google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu, a jax2tf export with
gradients disabled) holds the Flax variables by name: the EfficientNet-B3 backbone (params + BatchNorm batch_stats,
~10.7M trainable) and a 14,795-class ProtoPNet head, which is dropped. The `perch_v2` (GPU) variant has bit-identical
weights but only runs on CUDA, so the CPU one is the default.

Checks, on real annotated clips (5 s around each annotation, resampled to 32 kHz), each against the SavedModel's own
outputs; any failure exits 1 without saving:
  1. frontend: PerchV2Frontend(waveform) vs `spectrogram`                    (max |diff| <= --spec-tol)
  2. backbone: PerchV2Backbone(SavedModel spectrogram) vs `embedding` / `spatial_embedding` (relative L2 <= --tol)
  3. end to end: frontend + backbone (waveform) vs `embedding`              (relative L2 <= --tol)
  4. gradients: one backward pass in train mode reaches every parameter, all finite
Writes {"state_dict", "meta"} to --out (load with perch_v2.load_perch_v2_backbone).

Run with the TF venv (it has torch too):
    /home/noah/perch_venv/bin/python scripts/perch/convert_perch_v2_to_torch.py
"""
import argparse
import datetime as dt
import sys
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
import torch
from scipy.signal import resample_poly

from encoder_pipeline.model_trainer.perch_v2 import (
    SAMPLE_RATE, WINDOW_SAMPLES, PerchV2Backbone, PerchV2Frontend, perch_v2_state_dict,
)

REPO = Path(__file__).resolve().parents[2]
KAGGLE_HANDLE = "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu"
DEFAULT_SAVED_MODEL = Path.home() / ".cache/kagglehub/models" / KAGGLE_HANDLE / "1"
DEFAULT_ANNOTATIONS = [REPO / f"data_raw/DORI/20261022_190549/annotations_bgcap10000_ft_{dep}.csv"
                       for dep in ("dori_venus_strait_of_georgia", "limekiln", "tekteksen")]
DEFAULT_OUT = REPO / "data/model_trainer/pretrained/perch_v2_backbone.pt"


def resolve_saved_model(path: Path | None) -> Path:
    if path is not None:
        return path
    if DEFAULT_SAVED_MODEL.exists():
        return DEFAULT_SAVED_MODEL
    import kagglehub  # perch_hoplite's dependency; downloads into the same cache
    return Path(kagglehub.model_download(KAGGLE_HANDLE))


def load_clip(path: str, begin_sec: float, duration: float) -> np.ndarray:
    """5 s centred on the annotation, clamped to the file, mono, resampled to 32 kHz, zero-padded to 160,000."""
    info = sf.info(path)
    sr, file_dur = info.samplerate, info.frames / info.samplerate
    t0 = max(0.0, min(begin_sec + duration / 2 - 2.5, max(0.0, file_dur - 5.0)))
    audio, _ = sf.read(path, start=int(t0 * sr), stop=int((t0 + 5.0) * sr))
    y = (audio[:, 0] if audio.ndim > 1 else audio).astype(np.float64)
    g = gcd(SAMPLE_RATE, sr)
    y = resample_poly(y, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return np.pad(y, (0, max(0, WINDOW_SAMPLES - len(y))))[:WINDOW_SAMPLES]


def validation_clips(annotations: list[Path], per_label: int, seed: int) -> tuple[np.ndarray, list[str]]:
    waves, names = [], []
    for csv in annotations:
        if not csv.exists():
            print(f"  (skipping missing {csv})")
            continue
        ann = pd.read_csv(csv, usecols=["LocalPath", "FileBeginSec", "Duration", "Labels"], low_memory=False)
        for label, rows in ann.groupby("Labels"):
            for _, r in rows.sample(min(per_label, len(rows)), random_state=seed).iterrows():
                try:
                    waves.append(load_clip(r.LocalPath, r.FileBeginSec, r.Duration))
                    names.append(f"{csv.stem.removeprefix('annotations_bgcap10000_ft_')}/{label}")
                except Exception as e:  # unreadable file: use the others
                    print(f"  (skipping {r.LocalPath}: {type(e).__name__})")
    if not waves:
        raise SystemExit("no validation clips could be loaded; pass --annotations")
    return np.stack(waves), names


def rel_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.reshape(len(a), -1).astype(np.float64), b.reshape(len(b), -1).astype(np.float64)
    return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--saved-model", type=Path, default=None, help=f"default: {DEFAULT_SAVED_MODEL} (downloaded if missing)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--annotations", type=Path, nargs="+", default=DEFAULT_ANNOTATIONS,
                        help="annotation CSVs (LocalPath, FileBeginSec, Duration, Labels) to draw validation clips from")
    parser.add_argument("--clips-per-label", type=int, default=2, help="per label per CSV")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-4, help="max relative L2 error of embeddings")
    parser.add_argument("--spec-tol", type=float, default=1e-4, help="max |diff| of the log-mel spectrogram")
    args = parser.parse_args()
    # perch_v2_cpu's XLA module only runs on CPU; TF can see the GPU once torch has loaded the CUDA libraries
    tf.config.set_visible_devices([], "GPU")

    saved_model = resolve_saved_model(args.saved_model)
    print(f"SavedModel: {saved_model}")
    tf_model = tf.saved_model.load(str(saved_model))  # keep a reference: the signature's variables are weakly held
    serving = tf_model.signatures["serving_default"]
    variables = {v.name.split(":")[0]: v.numpy() for v in serving.variables}
    print(f"  {len(variables)} variables")

    backbone = PerchV2Backbone(input_layout="time_freq")
    backbone.load_state_dict(perch_v2_state_dict(variables, backbone))
    backbone.eval()
    frontend = PerchV2Frontend(output_layout="time_freq")
    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"  mapped onto PerchV2Backbone: {n_params:,} parameters")

    print("loading validation clips")
    waves, names = validation_clips(args.annotations, args.clips_per_label, args.seed)
    print(f"  {len(waves)} clips: " + ", ".join(f"{n} x{names.count(n)}" for n in dict.fromkeys(names)))

    ref = {k: [] for k in ("spectrogram", "embedding", "spatial_embedding")}
    for i in range(0, len(waves), 8):
        out = serving(inputs=tf.constant(waves[i:i + 8]))
        for k in ref:
            ref[k].append(out[k].numpy())
    ref = {k: np.concatenate(v) for k, v in ref.items()}

    with torch.no_grad():
        wave_t = torch.from_numpy(waves)
        spec = frontend(wave_t)  # (B, 1, 500, 128)
        spatial_from_ref = backbone.spatial(torch.from_numpy(ref["spectrogram"]).unsqueeze(1))
        emb_from_ref = spatial_from_ref.mean(dim=(2, 3)).numpy()
        emb_end_to_end = backbone(spec).numpy()
    spatial_from_ref = spatial_from_ref.permute(0, 2, 3, 1).numpy()  # -> (B, 16, 4, 1536) as the SavedModel

    checks = {
        "frontend spectrogram max|diff|": (float(np.abs(spec.squeeze(1).numpy() - ref["spectrogram"]).max()), args.spec_tol),
        "backbone embedding rel L2": (float(rel_l2(emb_from_ref, ref["embedding"]).max()), args.tol),
        "backbone spatial_embedding rel L2": (float(rel_l2(spatial_from_ref, ref["spatial_embedding"]).max()), args.tol),
        "end-to-end embedding rel L2": (float(rel_l2(emb_end_to_end, ref["embedding"]).max()), args.tol),
    }

    backbone.train()
    x = torch.from_numpy(ref["spectrogram"][:8]).unsqueeze(1)
    head = torch.randn(PerchV2Backbone.out_features, generator=torch.Generator().manual_seed(0)) * 0.01
    torch.nn.functional.softplus(-(backbone(x) @ head)).mean().backward()
    grads = [p.grad for p in backbone.parameters()]
    n_ok = sum(g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0 for g in grads)
    checks["parameters with finite non-zero grads"] = (n_ok, len(grads))
    backbone.load_state_dict(perch_v2_state_dict(variables, backbone))  # undo the train-mode BatchNorm stat update
    backbone.zero_grad(set_to_none=True)
    backbone.eval()

    failed = False
    print("\nchecks against the SavedModel:")
    for name, (value, limit) in checks.items():
        ok = value == limit if name.startswith("parameters") else value <= limit
        failed |= not ok
        shown = f"{value} / {limit}" if name.startswith("parameters") else f"{value:.2e} (limit {limit:.0e})"
        print(f"  {'PASS' if ok else 'FAIL'}  {name:40s} {shown}")

    if torch.cuda.is_available():  # informational: GPU float32 (TF32 off) vs the SavedModel
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
        with torch.no_grad():
            emb_gpu = backbone.cuda()(frontend.cuda()(wave_t.cuda())).cpu().numpy()
        backbone.cpu()
        print(f"  info  end-to-end embedding rel L2 on {torch.cuda.get_device_name()} (TF32 off): "
              f"{rel_l2(emb_gpu, ref['embedding']).max():.2e}")

    if failed:
        sys.exit("\nconversion check FAILED, nothing written")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"source_saved_model": str(saved_model), "converted_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "tensorflow": tf.__version__, "torch": str(torch.__version__), "n_parameters": n_params,
            "n_validation_clips": len(waves),
            **{k: float(v) for k, (v, _) in checks.items() if not k.startswith("parameters")}}
    torch.save({"state_dict": backbone.state_dict(), "meta": meta}, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
