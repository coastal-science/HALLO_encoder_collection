# Per-class score thresholds for the TIDAS species probe (configs/tidas/01_perch_v2_species_probe_5fold.yaml) at a
# target precision, from its out-of-fold predictions. Each fold's val split is out-of-fold, so the 5 val splits
# together score every non-test file once: thresholds are picked there. The fixed test files are then scored by
# the mean of the 5 fold probes (the ensemble that labels the pull) and the thresholds are checked on them.
# One-vs-rest per class: a window counts for class c when prob_c >= threshold_c.
# Writes <embeddings dir>/thresholds_p<target>.json, used by the TIDAS inference.
#   /home/noah/embed_venv/bin/python scripts/tidas/probe_thresholds.py data/embeddings/<run_id> [--precision 0.95]

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def threshold_at_precision(score: np.ndarray, positive: np.ndarray, target: float) -> tuple[float, float, float]:
    """Lowest threshold whose precision is >= target (most recall at that precision): (threshold, precision, recall).
    NaN threshold when no threshold reaches the target."""
    order = np.argsort(-score)
    s, p = score[order], positive[order]
    tp = np.cumsum(p)
    precision = tp / np.arange(1, len(p) + 1)
    recall = tp / max(1, p.sum())
    # only cut between distinct scores
    last_of_tie = np.r_[s[1:] != s[:-1], True]
    ok = np.where((precision >= target) & last_of_tie)[0]
    if len(ok) == 0:
        return float("nan"), float("nan"), 0.0
    i = ok[-1]
    return float(s[i]), float(precision[i]), float(recall[i])


def apply(score, positive, t):
    hit = score >= t
    return (float(positive[hit].mean()) if hit.any() else float("nan")), float(hit[positive].mean()) if positive.any() else float("nan"), int(hit.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("embeddings_dir", type=Path)
    ap.add_argument("--precision", type=float, default=0.95)
    a = ap.parse_args()
    frames = [pd.read_csv(f).assign(fold=int(f.name[4])) for f in sorted(a.embeddings_dir.glob("fold*_linear_probe_predictions.csv"))]
    if not frames:
        raise SystemExit(f"no fold*_linear_probe_predictions.csv in {a.embeddings_dir}")
    df = pd.concat(frames)
    classes = [c[len("prob_"):] for c in df.columns if c.startswith("prob_")]
    val = df[df.split == "val"]
    test = df[df.split == "test"].groupby("uid").agg({"true_label": "first", **{f"prob_{c}": "mean" for c in classes}})
    out, rows = {"target_precision": a.precision, "folds": len(frames), "classes": {}}, []
    for c in classes:
        if c == "Background":
            continue
        t, p, r = threshold_at_precision(val[f"prob_{c}"].to_numpy(), (val.true_label == c).to_numpy(), a.precision)
        tp_, tr_, n_hit = apply(test[f"prob_{c}"].to_numpy(), (test.true_label == c).to_numpy(), t) if t == t else (float("nan"), 0.0, 0)
        out["classes"][c] = {"threshold": t, "val_precision": p, "val_recall": r, "test_precision": tp_,
                             "test_recall": tr_, "test_hits": n_hit, "val_n": int((val.true_label == c).sum()),
                             "test_n": int((test.true_label == c).sum())}
        rows.append((c, t, p, r, tp_, tr_, int((val.true_label == c).sum()), int((test.true_label == c).sum())))
    # "any marine mammal": 1 - P(Background), one threshold for presence of any class
    any_val = 1 - val["prob_Background"].to_numpy()
    t, p, r = threshold_at_precision(any_val, (val.true_label != "Background").to_numpy(), a.precision)
    tp_, tr_, n_hit = apply(1 - test["prob_Background"].to_numpy(), (test.true_label != "Background").to_numpy(), t)
    out["any_marine_mammal"] = {"threshold": t, "val_precision": p, "val_recall": r, "test_precision": tp_, "test_recall": tr_}
    rows.append(("ANY (1 - P(Background))", t, p, r, tp_, tr_, int((val.true_label != "Background").sum()),
                 int((test.true_label != "Background").sum())))
    path = a.embeddings_dir / f"thresholds_p{int(round(a.precision * 100))}.json"
    path.write_text(json.dumps(out, indent=2))
    table = pd.DataFrame(rows, columns=["class", "threshold", "val_P", "val_R", "test_P", "test_R", "val_n", "test_n"])
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
