"""Paired bootstrap of SRKW-vs-rest test precision / recall / F1 differences between methods on
exp_3's test clips (zero-shot / AdaBN from adabn_exp3.py, exp_3's linear probe / fine-tune).

Per site, model and decision rule (argmax, val_tuned -- see plot_exp3_comparison.py), each method
has one SRKW decision per fold per test clip. A bootstrap replicate reweights the test clips and
scores every method on the SAME replicate (paired); a method's score is its mean over the 5 folds,
matching the reported fold means. Two resampling units:
  - file: whole recording files (LocalPath) drawn with replacement -- clips from one file are
          correlated, so this is the honest one
  - clip: individual clips drawn with replacement -- narrower, shown for contrast

Writes bootstrap_differences.csv: point estimate (full test set), 95% percentile CI and
p_le_0 = share of replicates where the difference is <= 0 (one-sided).

    python scripts/adaptation/bootstrap_exp3.py [--n-boot 10000]
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from adabn import REPO
from plot_exp3_comparison import exp3_predictions, srkw_threshold

EXP2_HDF5 = REPO / "data/preprocessor/43af4e010f9c6f92_20261022_191758.h5"  # exp_2's dataset: uid -> LocalPath
COMPARISONS = [  # (method, baseline)
    ("adabn_train", "zeroshot"),
    ("adabn_balanced", "zeroshot"),
    ("adabn_balanced_srkw_rest", "zeroshot"),
    ("linear_probe", "zeroshot"),
    ("finetune", "zeroshot"),
    ("adabn_train", "finetune"),
    ("adabn_balanced", "finetune"),
    ("adabn_balanced_srkw_rest", "finetune"),
]
METRICS = ["precision", "recall", "f1"]
SEED = 0


def decisions(preds: pd.DataFrame, test_uids: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """{rule: {method: bool[fold, clip]}} of SRKW decisions on test_uids, per fold."""
    out: dict[str, dict[str, np.ndarray]] = {"argmax": {}, "val_tuned": {}}
    for method, by_method in preds.groupby("method"):
        argmax, tuned = [], []
        for _, pred in sorted(by_method.groupby("fold"), key=lambda kv: kv[0]):
            test = pred[pred.split == "test"].set_index("uid").loc[test_uids]
            threshold = srkw_threshold(pred[pred.split == "val"])
            argmax.append((test.pred_label == "SRKW").to_numpy())
            tuned.append((test.prob_SRKW >= threshold).to_numpy())
        out["argmax"][method], out["val_tuned"][method] = np.array(argmax), np.array(tuned)
    return out


def fold_mean_scores(weights: np.ndarray, y: np.ndarray, d: np.ndarray) -> dict[str, np.ndarray]:
    """Fold-mean precision / recall / F1 for each row of weights (replicates x clips); d is folds x clips."""
    tp = weights @ (d & y).T.astype(float)  # replicates x folds
    fp = weights @ (d & ~y).T.astype(float)
    fn = weights @ (~d & y).T.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.nan_to_num(tp / (tp + fp))
        recall = np.nan_to_num(tp / (tp + fn))
        f1 = np.nan_to_num(2 * tp / (2 * tp + fp + fn))
    return {"precision": precision.mean(axis=1), "recall": recall.mean(axis=1), "f1": f1.mean(axis=1)}


def replicate_weights(groups: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """replicates x clips weights: each replicate draws len(unique groups) groups with replacement and
    weights each clip by how often its group was drawn."""
    codes, uniques = pd.factorize(groups)
    counts = rng.multinomial(len(uniques), np.full(len(uniques), 1 / len(uniques)), size=n_boot)
    return counts[:, codes].astype(float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--adabn-predictions", type=Path, default=REPO / "scripts/adaptation/results/exp3_splits/adabn_exp3_predictions.csv")
    parser.add_argument("--out", type=Path, default=REPO / "scripts/adaptation/results/exp3_splits/bootstrap_differences.csv")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)

    preds = pd.concat([pd.read_csv(args.adabn_predictions).rename(columns={"variant": "method"}), exp3_predictions()])
    with h5py.File(EXP2_HDF5, "r") as h5:
        file_of = pd.Series(h5["LocalPath"].asstr()[:], index=h5["uid"][:])

    rows = []
    for (model, site), site_preds in preds.groupby(["model", "site"]):
        test = site_preds[site_preds.split == "test"]
        test_uids = np.sort(test.uid.unique())
        y = (test.drop_duplicates("uid").set_index("uid").loc[test_uids].true_label == "SRKW").to_numpy()
        files = file_of.loc[test_uids].to_numpy()
        units = {
            "file": replicate_weights(files, args.n_boot, rng),
            "clip": replicate_weights(np.arange(len(test_uids)), args.n_boot, rng),
        }
        print(f"{model} {site}: {len(test_uids)} test clips ({y.sum()} SRKW) from {len(np.unique(files))} files", flush=True)
        for rule, by_method in decisions(site_preds, test_uids).items():
            full = {m: fold_mean_scores(np.ones((1, len(y))), y, d) for m, d in by_method.items()}
            for unit, weights in units.items():
                boot = {m: fold_mean_scores(weights, y, d) for m, d in by_method.items()}
                for method, baseline in COMPARISONS:
                    if method not in by_method or baseline not in by_method:
                        continue
                    for metric in METRICS:
                        diff = boot[method][metric] - boot[baseline][metric]
                        lo, hi = np.percentile(diff, [2.5, 97.5])
                        rows.append(dict(
                            model=model, site=site, rule=rule, unit=unit, method=method, baseline=baseline, metric=metric,
                            method_score=full[method][metric][0], baseline_score=full[baseline][metric][0],
                            diff=full[method][metric][0] - full[baseline][metric][0], ci_low=lo, ci_high=hi,
                            p_le_0=float((diff <= 0).mean()),
                        ))
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
