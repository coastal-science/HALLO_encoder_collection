"""SRKW-vs-rest test precision / recall / F1 over exp_3's 5 folds: exp_2 zero-shot and AdaBN variants
(adabn_exp3.py) next to exp_3's linear probe and fine-tune, one panel per site, one figure per
exp_2 backbone and decision rule. All methods are scored on the same exp_3 test clips; exp_3 only
fine-tuned effnet_b1, so the resnet18 figures have no linear probe / fine-tune boxes.

Two decision rules, both computed from per-clip predictions:
  - argmax:    SRKW iff it is the 4-class argmax (the default operating point)
  - val_tuned: SRKW iff prob_SRKW >= t, with t the threshold maximising SRKW-vs-rest F1 on that
               fold's val split (no test labels used), per method / site / fold

Reads scripts/adaptation/results/exp3_splits/adabn_exp3_predictions.csv and exp_3's fine-tune /
linear-probe predictions from MLflow; writes srkw_comparison_metrics.csv next to the figures.

    python scripts/adaptation/plot_exp3_comparison.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score

REPO = Path(__file__).resolve().parents[2]
TRACKING_URI = "http://localhost:5000"
EXP3_PREFIX = "exp_3_effnet_b1_ft_"
N_FOLDS = 5
SITES = {"strait_of_georgia": "Strait of Georgia", "limekiln": "Limekiln", "tekteksen": "Tekteksen"}
METHODS = {  # method -> x tick label
    "zeroshot": "zero-shot",
    "adabn_train": "AdaBN\ntrain mix",
    "adabn_balanced": "AdaBN\nbalanced*",
    "adabn_balanced_srkw_rest": "AdaBN\n50/50*",
    "linear_probe": "linear\nprobe*",
    "finetune": "fine-tune*",
}
METRICS = {  # metric -> (label, colour)
    "precision": ("Precision", "#2a78d6"), "recall": ("Recall", "#eb6834"), "f1": ("F1", "#1baf7a"),
}
RULES = {  # rule -> title fragment
    "argmax": "default argmax decision (* uses target-site labels)",
    "val_tuned": "threshold tuned on each fold's labelled val split (every method uses target-site val labels; "
                 "* also uses train labels)",
}


def exp3_predictions() -> pd.DataFrame:
    """val + test predictions of exp_3's fine-tune (parent run) and linear probe (embeddings_foldN child run)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("dori_experiments")
    frames = []
    for run in client.search_runs([exp.experiment_id], f"attributes.run_name LIKE '{EXP3_PREFIX}%'"):
        site = run.info.run_name.removeprefix(EXP3_PREFIX).removesuffix("_5fold")
        children = {
            child.info.run_name: child.info.run_id
            for child in client.search_runs([exp.experiment_id], f"tags.mlflow.parentRunId = '{run.info.run_id}'")
        }
        for fold in range(N_FOLDS):
            sources = {
                "finetune": (run.info.run_id, f"fold{fold}_predictions.csv"),
                "linear_probe": (children[f"embeddings_fold{fold}"], f"fold{fold}_linear_probe_predictions.csv"),
            }
            for method, (run_id, artifact) in sources.items():
                frame = pd.read_csv(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact))
                frames.append(frame.assign(model="effnet_b1", site=site, fold=fold, method=method))
    return pd.concat(frames)


def srkw_threshold(val: pd.DataFrame) -> float:
    """prob_SRKW threshold maximising SRKW-vs-rest F1 on val."""
    precision, recall, thresholds = precision_recall_curve((val.true_label == "SRKW").to_numpy(), val.prob_SRKW)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[np.argmax(f1)])


def srkw_scores(pred: pd.DataFrame) -> list[dict]:
    """Test SRKW-vs-rest precision / recall / F1 under each decision rule, threshold picked on val."""
    test = pred[pred.split == "test"]
    y_test = (test.true_label == "SRKW").to_numpy()
    threshold = srkw_threshold(pred[pred.split == "val"])
    decisions = {
        "argmax": (test.pred_label == "SRKW").to_numpy(),
        "val_tuned": (test.prob_SRKW >= threshold).to_numpy(),
    }
    return [
        dict(rule=rule, threshold=threshold if rule == "val_tuned" else np.nan, metric=name, value=fn(y_test, p, zero_division=0))
        for rule, p in decisions.items()
        for name, fn in [("precision", precision_score), ("recall", recall_score), ("f1", f1_score)]
    ]


def load(adabn_predictions: Path) -> pd.DataFrame:
    adabn = pd.read_csv(adabn_predictions).rename(columns={"variant": "method"})
    preds = pd.concat([adabn, exp3_predictions()])
    rows = []
    for (model, site, fold, method), pred in preds.groupby(["model", "site", "fold", "method"]):
        rows += [dict(model=model, site=site, fold=fold, method=method, **r) for r in srkw_scores(pred)]
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, model: str, rule: str, out: Path) -> None:
    df = df[df.rule == rule]
    methods = [m for m in METHODS if not df.query("model == @model and method == @m").empty]
    fig, axes = plt.subplots(1, len(SITES), figsize=(19, 4.8), sharey=True)
    width = 0.25
    offsets = {m: (i - (len(METRICS) - 1) / 2) * width * 1.15 for i, m in enumerate(METRICS)}
    for ax, (site, label) in zip(axes, SITES.items()):
        for xi, method in enumerate(methods):
            for metric, (_, colour) in METRICS.items():
                vals = df.query("model == @model and site == @site and method == @method and metric == @metric").value
                x = xi + offsets[metric]
                ax.boxplot(
                    vals, positions=[x], widths=width, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colour + "40", edgecolor=colour, linewidth=1.5),
                    medianprops=dict(color=colour, linewidth=2),
                    whiskerprops=dict(color=colour, linewidth=1.2),
                    capprops=dict(color=colour, linewidth=1.2),
                )
                ax.scatter([x] * len(vals), vals, s=16, color=colour, edgecolor="white", linewidth=0.8, zorder=3)
        if "linear_probe" in methods:  # AdaBN (BN stats only) | trained heads
            ax.axvline(methods.index("linear_probe") - 0.5, color="#b5b4ad", linewidth=0.8, linestyle=":")
        ax.set_title(label, loc="left", fontsize=11)
        ax.set_xticks(range(len(methods)), [METHODS[m] for m in methods], fontsize=8.5)
        ax.set_xlim(-0.6, len(methods) - 0.4)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#b5b4ad")
        ax.spines["bottom"].set_color("#b5b4ad")
    axes[0].set_ylabel("SRKW test score")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c + "40", edgecolor=c, linewidth=1.5) for _, c in METRICS.values()]
    fig.legend(handles, [name for name, _ in METRICS.values()], loc="upper center", ncol=len(METRICS), frameon=False,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"exp_2 {model} adapted to each site vs exp_3, SRKW vs rest on exp_3's test clips, 5 folds, "
                 f"{RULES[rule]}", x=0.01, ha="left", y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adabn-predictions", type=Path, default=REPO / "scripts/adaptation/results/exp3_splits/adabn_exp3_predictions.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO / "scripts/adaptation/results/exp3_splits")
    args = parser.parse_args()
    df = load(args.adabn_predictions)
    df.to_csv(args.out_dir / "srkw_comparison_metrics.csv", index=False)
    for model in sorted(df.model.unique()):
        for rule in RULES:
            plot(df, model, rule, args.out_dir / f"srkw_comparison_{model}_{rule}.png")
