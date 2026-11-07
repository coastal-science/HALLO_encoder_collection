"""Experiment E: SRKW-vs-rest test precision-recall curves per site (rows) and label-budget tier (columns).

Same runs, folds and test clips as compare_label_budget.py. Each arm's curve is the mean over folds of its
interpolated precision (max precision at recall >= r) on a common recall grid; its legend entry gives the mean
PR-AUC (average precision) over folds. The exp_2 effnet_b1 zero-shot model is the dashed grey reference and the
dotted line is SRKW prevalence on the test clips (a no-skill detector).

    python scripts/adaptation/plot_label_budget_pr_curves.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score, precision_recall_curve

from compare_label_budget import ARMS, FRACS, budget_prefixes
from compare_pretraining_results import REPO, SITES, TRACKING_URI, comparison_predictions, restrict_to_folds

RECALL = np.linspace(0, 1, 201)
INK, MUTED, GRID, REF = "#0b0b0b", "#52514e", "#e4e3dd", "#898781"


def curve(test: pd.DataFrame) -> tuple[np.ndarray, float]:
    """interpolated precision on RECALL and average precision of one fold's test predictions."""
    y = (test.true_label == "SRKW").to_numpy()
    precision, recall, _ = precision_recall_curve(y, test.prob_SRKW)
    envelope = np.maximum.accumulate(precision)  # max precision at recall >= r; recall is decreasing
    return np.interp(RECALL, recall[::-1], envelope[::-1]), average_precision_score(y, test.prob_SRKW)


def mean_curve(preds: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    folds = [curve(p[p.split == "test"]) for _, p in preds.groupby("fold")]
    aps = [ap for _, ap in folds]
    return np.mean([c for c, _ in folds], axis=0), float(np.mean(aps)), float(np.std(aps))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adabn-predictions", type=Path,
                        default=REPO / "scripts/adaptation/results/exp3_splits/adabn_exp3_predictions.csv")
    parser.add_argument("--out", type=Path,
                        default=REPO / "scripts/adaptation/results/pretraining_comparison/budget_pr_curves.png")
    args = parser.parse_args()
    mlflow.set_tracking_uri(TRACKING_URI)

    missing: list[str] = []
    preds = restrict_to_folds(comparison_predictions(missing, budget_prefixes()))
    tagged = preds.method.str.split("@", expand=True)
    preds = preds.assign(method=tagged[0], frac=tagged[1].astype(int))
    adabn = pd.read_csv(args.adabn_predictions).rename(columns={"variant": "method"})
    zeroshot = restrict_to_folds(adabn[(adabn.method == "zeroshot") & (adabn.model == "effnet_b1")])

    tiers = FRACS + [100]
    fig, axes = plt.subplots(len(SITES), len(tiers), figsize=(3.6 * len(tiers), 3.5 * len(SITES)), sharex=True, sharey=True)
    for row, (site, (_, site_label)) in zip(axes, SITES.items()):
        z = zeroshot[zeroshot.site == site]
        z_curve, z_ap, _ = mean_curve(z)
        test = z[(z.split == "test") & (z.fold == z.fold.min())]
        prevalence = (test.true_label == "SRKW").mean()
        for col, (ax, frac) in enumerate(zip(row, tiers)):
            ax.axhline(prevalence, color=REF, linewidth=1, linestyle=":", zorder=1)
            ax.plot(RECALL, z_curve, color=REF, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
            handles = [Line2D([], [], color=REF, linewidth=1.5, linestyle=(0, (4, 3)), label=f"{z_ap:.2f}")]
            for method, (label, colour) in ARMS.items():
                p = preds[(preds.site == site) & (preds.method == method) & (preds.frac == frac)]
                if p.empty:
                    continue
                c, ap, sd = mean_curve(p)
                ax.plot(RECALL, c, color=colour, linewidth=2, zorder=3)
                handles.append(Line2D([], [], color=colour, linewidth=2, label=f"{ap:.2f} ± {sd:.2f}"))
            ax.legend(handles=handles, title="PR-AUC", title_fontsize=8, fontsize=8, loc="lower left", frameon=True,
                      facecolor="white", edgecolor="none", framealpha=0.85, labelcolor=INK, handlelength=1.4)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.grid(color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=9)
            if ax is axes[0, col]:
                ax.set_title(f"{frac}% of site labels", fontsize=11, color=INK)
            if col == 0:
                ax.set_ylabel(f"{site_label}\nprecision", color=INK)
            if ax is axes[-1, col]:
                ax.set_xlabel("recall", color=INK)

    legend = [Line2D([], [], color=c, linewidth=2, label=l) for l, c in ARMS.values()]
    legend += [Line2D([], [], color=REF, linewidth=1.5, linestyle=(0, (4, 3)), label="zero-shot effnet_b1 (no site labels)"),
               Line2D([], [], color=REF, linewidth=1, linestyle=":", label="SRKW prevalence (no skill)")]
    fig.legend(handles=legend, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("SRKW vs rest test PR curves by label budget, mean over 5 folds",
                 x=0.01, ha="left", y=0.995, fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")
    for m in missing:
        print(f"missing: {m}")


if __name__ == "__main__":
    main()
