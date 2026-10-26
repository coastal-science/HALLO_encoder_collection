"""Boxplots of exp_3's 5-fold test scores collapsed to SRKW vs Background, one panel per location (1x3).

Recomputes test precision / recall from each fold's predictions in MLflow, for two heads:
  - fine-tune:    foldN_predictions.csv on the exp_3 parent run
  - linear probe: foldN_linear_probe_predictions.csv on its embeddings_foldN child run
Every non-SRKW class (Background, HW, TKW) is merged into Background, and the 4-class argmax is
kept as the prediction (SRKW iff pred_label == SRKW).

    python scripts/DORI/plot_exp3_kfold_srkw_binary.py [--out-dir configs/experiments/exp_3]
"""

import argparse

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import precision_score, recall_score

TRACKING_URI = "http://localhost:5000"
RUN_PREFIX = "exp_3_effnet_b1_ft_"
N_FOLDS = 5
CLASSES = ["Background", "SRKW"]
LOCATIONS = {  # run-name slug -> label
    "strait_of_georgia": "Strait of Georgia",
    "limekiln": "Limekiln",
    "tekteksen": "Tekteksen",
}
METRICS = {"precision": ("Precision", "#2a78d6"), "recall": ("Recall", "#eb6834")}  # metric -> (label, colour)
HEADS = {  # head -> (title, predictions artifact; {fold} filled in)
    "finetune": ("EfficientNet-B1 fine-tune", "fold{fold}_predictions.csv"),
    "linear_probe": ("linear probe on fine-tuned embeddings", "fold{fold}_linear_probe_predictions.csv"),
}


def binary_scores(pred: pd.DataFrame) -> list[dict]:
    test = pred[pred.split == "test"]
    y_true = (test.true_label == "SRKW").astype(int)
    y_pred = (test.pred_label == "SRKW").astype(int)
    rows = []
    for cls, pos in [("SRKW", 1), ("Background", 0)]:
        t, p = (y_true == pos).astype(int), (y_pred == pos).astype(int)
        rows.append(dict(cls=cls, metric="precision", value=precision_score(t, p, zero_division=0), n_pos=int(t.sum())))
        rows.append(dict(cls=cls, metric="recall", value=recall_score(t, p, zero_division=0), n_pos=int(t.sum())))
    return rows


def load_metrics() -> pd.DataFrame:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("dori_experiments")
    rows = []
    for run in client.search_runs([exp.experiment_id], f"attributes.run_name LIKE '{RUN_PREFIX}%'"):
        loc = run.info.run_name.removeprefix(RUN_PREFIX).removesuffix("_5fold")
        children = {
            child.info.run_name: child.info.run_id
            for child in client.search_runs([exp.experiment_id], f"tags.mlflow.parentRunId = '{run.info.run_id}'")
        }
        for fold in range(N_FOLDS):
            run_ids = {"finetune": run.info.run_id, "linear_probe": children[f"embeddings_fold{fold}"]}
            for head, (_, artifact) in HEADS.items():
                path = mlflow.artifacts.download_artifacts(run_id=run_ids[head], artifact_path=artifact.format(fold=fold))
                for row in binary_scores(pd.read_csv(path)):
                    rows.append(dict(head=head, location=loc, fold=fold, **row))
    return pd.DataFrame(rows)


def style(ax: plt.Axes) -> None:
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_xlim(-0.6, len(CLASSES) - 0.4)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#b5b4ad")
    ax.spines["bottom"].set_color("#b5b4ad")


def plot(df: pd.DataFrame, head: str, out: str) -> None:
    fig, axes = plt.subplots(1, len(LOCATIONS), figsize=(14, 4.8), sharey=True)
    width = 0.3
    offsets = {m: (i - 0.5) * width * 1.2 for i, m in enumerate(METRICS)}
    for ax, (loc, label) in zip(axes, LOCATIONS.items()):
        for ci, cls in enumerate(CLASSES):
            for metric, (_, colour) in METRICS.items():
                x = ci + offsets[metric]
                vals = df.query("head == @head and location == @loc and metric == @metric and cls == @cls").value
                ax.boxplot(
                    vals, positions=[x], widths=width, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colour + "40", edgecolor=colour, linewidth=1.5),
                    medianprops=dict(color=colour, linewidth=2),
                    whiskerprops=dict(color=colour, linewidth=1.2),
                    capprops=dict(color=colour, linewidth=1.2),
                )
                ax.scatter([x] * len(vals), vals, s=18, color=colour, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_title(label, loc="left", fontsize=11)
        style(ax)
    axes[0].set_ylabel("test score")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c + "40", edgecolor=c, linewidth=1.5) for _, c in METRICS.values()]
    fig.legend(handles, [name for name, _ in METRICS.values()], loc="upper center", ncol=len(METRICS), frameon=False,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"exp_3 {HEADS[head][0]}: SRKW vs Background (HW, TKW merged into Background), 5 folds",
                 x=0.01, ha="left", y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="configs/experiments/exp_3")
    args = parser.parse_args()
    df = load_metrics()
    df.sort_values(["head", "location", "metric", "cls", "fold"]).to_csv(
        f"{args.out_dir}/exp_3_kfold_srkw_binary_metrics.csv", index=False
    )
    for head in HEADS:
        plot(df, head, f"{args.out_dir}/exp_3_kfold_boxplot_srkw_binary_{head}.png")
