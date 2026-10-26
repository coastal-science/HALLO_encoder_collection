"""Boxplots of exp_3's 5-fold per-class test metrics, one figure per DORI fine-tune location.

Pulls foldN_test_<metric>_<class> from the exp_3 parent runs in MLflow. A class with no
examples at a location (HW at Strait of Georgia, TKW at Limekiln / Tekteksen) logs 0.0
there; it is left out of the plot and marked "absent" instead of drawn as a 0 box.

    python scripts/DORI/plot_exp3_kfold_boxplot.py [--out-dir configs/experiments/exp_3]
"""

import argparse
import re

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

TRACKING_URI = "http://localhost:5000"
RUN_PREFIX = "exp_3_effnet_b1_ft_"
CLASSES = ["Background", "HW", "SRKW", "TKW"]
LOCATIONS = {  # run-name slug -> label
    "strait_of_georgia": "Strait of Georgia",
    "limekiln": "Limekiln",
    "tekteksen": "Tekteksen",
}
ABSENT = {("strait_of_georgia", "HW"), ("limekiln", "TKW"), ("tekteksen", "TKW")}
METRICS = {"f1": ("F1", "#2a78d6"), "pr_auc": ("PR-AUC", "#eb6834")}  # metric -> (label, colour)
KEY = re.compile(r"fold(\d+)_test_(f1|pr_auc|precision|recall)_(\w+)$")


def load_metrics() -> pd.DataFrame:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("dori_experiments")
    rows = []
    for run in client.search_runs([exp.experiment_id], f"attributes.run_name LIKE '{RUN_PREFIX}%'"):
        loc = run.info.run_name.removeprefix(RUN_PREFIX).removesuffix("_5fold")
        for key, value in run.data.metrics.items():
            if m := KEY.match(key):
                rows.append(dict(location=loc, fold=int(m[1]), metric=m[2], cls=m[3], value=value))
    return pd.DataFrame(rows)


def plot_location(df: pd.DataFrame, loc: str, out: str) -> None:
    label = LOCATIONS[loc]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    width = 0.3
    offsets = {m: (i - 0.5) * width * 1.2 for i, m in enumerate(METRICS)}
    for ci, cls in enumerate(CLASSES):
        if (loc, cls) in ABSENT:
            ax.text(ci, 0.03, "absent at this site", ha="center", va="bottom", fontsize=8, color="#8a8a85")
            continue
        for metric, (_, colour) in METRICS.items():
            x = ci + offsets[metric]
            vals = df.query("location == @loc and metric == @metric and cls == @cls").value
            ax.boxplot(
                vals, positions=[x], widths=width, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=colour + "40", edgecolor=colour, linewidth=1.5),
                medianprops=dict(color=colour, linewidth=2),
                whiskerprops=dict(color=colour, linewidth=1.2),
                capprops=dict(color=colour, linewidth=1.2),
            )
            ax.scatter([x] * len(vals), vals, s=18, color=colour, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_xlim(-0.6, len(CLASSES) - 0.4)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("test score")
    ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#b5b4ad")
    ax.spines["bottom"].set_color("#b5b4ad")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c + "40", edgecolor=c, linewidth=1.5) for _, c in METRICS.values()]
    ax.legend(handles, [name for name, _ in METRICS.values()], loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=len(METRICS), frameon=False)
    fig.suptitle(f"exp_3 {label}: EfficientNet-B1 fine-tune, per-class test scores over 5 folds",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="configs/experiments/exp_3")
    args = parser.parse_args()
    df = load_metrics()
    df.sort_values(["metric", "location", "cls", "fold"]).to_csv(f"{args.out_dir}/exp_3_kfold_metrics.csv", index=False)
    for loc in LOCATIONS:
        plot_location(df, loc, f"{args.out_dir}/exp_3_kfold_boxplot_{loc}.png")
