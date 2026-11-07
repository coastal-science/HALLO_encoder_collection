"""How close the SSL approach (SSAMBA-tiny C) gets to the supervised one (EffNet-B1 fine-tune), SRKW vs rest, per
deployment. Only these two are shown.

Left column: mean score per site of both. Right column: SSAMBA minus EffNet per fold (fold i of
both is trained / tested on the same split, so differences are paired), mean ± 95% t-interval, with a shaded
±MARGIN band. Rows: PR-AUC (threshold-free) and F1 at the val-tuned threshold.

Reads metrics_per_fold.csv from compare_pretraining_results.py, so run that first; SSAMBA arms without results yet
are skipped.

    python scripts/adaptation/plot_gap_to_supervised.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from compare_pretraining_results import SITES

REPO = Path(__file__).resolve().parents[2]
SUPERVISED = "effnet_b1_ft"
BASELINES = {}  # method -> (label, marker); none shown
ARMS = {  # method -> (label, colour); reference palette slots 1 and 2
    SUPERVISED: ("SL: EffNet-B1, supervised pretraining", "#2a78d6"),
    "ssamba_C_ssl_all": ("SSL: SSAMBA-tiny, self-supervised pretraining", "#eb6834"),
}
METRICS = {("any", "pr_auc"): "PR-AUC", ("val_tuned", "f1"): "F1, val-tuned threshold"}
MARGIN = 0.05
INK, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3dd", "#f0efea"


def style(ax) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, left=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPO / "scripts/adaptation/results/pretraining_comparison")
    args = parser.parse_args()
    df = pd.read_csv(args.out_dir / "metrics_per_fold.csv")
    arms = [m for m in ARMS if m in set(df.method)]
    ssamba = [m for m in arms if m != SUPERVISED]
    sites = list(SITES)
    y = {site: len(sites) - 1 - i for i, site in enumerate(sites)}
    step = 0.5 / max(len(ssamba), 1)

    fig, axes = plt.subplots(len(METRICS), 2, figsize=(14, 4 + 2.2 * len(METRICS)), gridspec_kw=dict(width_ratios=[1.1, 1]))
    rows = []
    for row, ((rule, metric), metric_label) in zip(axes, METRICS.items()):
        sel = df[(df.rule == rule) & (df.metric == metric)]
        ladder, gap = row
        for site in sites:
            s = sel[sel.site == site]
            means = s.groupby("method").value.mean()
            lo, hi = means.reindex(list(BASELINES) + arms).dropna().agg(["min", "max"])
            ladder.plot([lo, hi], [y[site]] * 2, color=GRID, linewidth=6, solid_capstyle="round", zorder=1)
            for method, (_, marker) in BASELINES.items():
                if method in means:
                    ladder.scatter(means[method], y[site], marker=marker, s=60, color=MUTED, linewidth=1.5,
                                   facecolor="white" if marker == "s" else MUTED, zorder=3)
            if SUPERVISED in means:  # hollow ring, so an SSAMBA arm that matches it shows as a dot inside it
                ladder.scatter(means[SUPERVISED], y[site], s=220, facecolor="white", edgecolor=ARMS[SUPERVISED][1],
                               linewidth=2.2, zorder=3)
            for method in ssamba:
                if method in means:
                    ladder.scatter(means[method], y[site], s=70, color=ARMS[method][1], edgecolor="white",
                                   linewidth=1.2, zorder=4)

            sup = s[s.method == SUPERVISED].set_index("fold").value
            for k, method in enumerate(ssamba):
                diff = (s[s.method == method].set_index("fold").value - sup).dropna()
                if diff.empty:
                    continue
                yy = y[site] + (k - (len(ssamba) - 1) / 2) * step
                half = stats.t.ppf(0.975, len(diff) - 1) * diff.std() / np.sqrt(len(diff)) if len(diff) > 1 else 0
                colour = ARMS[method][1]
                gap.scatter(diff, [yy] * len(diff), s=22, color=colour, alpha=0.55, edgecolor="white", linewidth=0.6,
                            zorder=3)
                gap.errorbar(diff.mean(), yy, xerr=half, fmt="o", color=colour, markersize=8, elinewidth=2,
                             capsize=4, markeredgecolor="white", zorder=4)
                rows.append(dict(metric=metric_label, site=SITES[site][1], method=method, mean_gap=diff.mean(),
                                 ci_low=diff.mean() - half, ci_high=diff.mean() + half, n_folds=len(diff)))

        ladder.set_yticks(list(y.values()), [SITES[s][1] for s in y], fontsize=10, color=INK)
        ladder.set_title(f"{metric_label}: mean over 5 folds", loc="left", fontsize=11, color=INK)
        ladder.set_xlabel(metric_label, color=MUTED)
        style(ladder)

        gap.axvspan(-MARGIN, MARGIN, color=BAND, zorder=0)
        gap.axvline(0, color=MUTED, linewidth=1, zorder=1)
        gap.set_yticks(list(y.values()), [SITES[s][1] for s in y], fontsize=10, color=INK)
        gap.set_ylim(-0.6, len(sites) - 0.4)
        ladder.set_ylim(-0.6, len(sites) - 0.4)
        gap.set_title(f"SSL − SL per fold (dots), mean ± 95% CI; band = ±{MARGIN}",
                      loc="left", fontsize=11, color=INK)
        gap.set_xlabel(f"Δ {metric_label} (positive = SSL better)", color=MUTED)
        style(gap)

    handles = ([plt.Line2D([], [], marker=m, linestyle="", color=MUTED, markerfacecolor="white" if m == "s" else MUTED,
                           markersize=8) for _, m in BASELINES.values()]
               + [plt.Line2D([], [], marker="o", linestyle="", markersize=12, markerfacecolor="white",
                             markeredgecolor=ARMS[SUPERVISED][1], markeredgewidth=2.2)]
               + [plt.Line2D([], [], marker="o", linestyle="", color=ARMS[m][1], markersize=8) for m in ssamba])
    fig.legend(handles, [l for l, _ in BASELINES.values()] + [ARMS[m][0] for m in [SUPERVISED] + ssamba], loc="upper center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.96), fontsize=9)
    fig.suptitle("SRKW vs rest: SSL vs SL pretraining, both fine-tuned on the same target-site folds", x=0.01, ha="left", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = args.out_dir / "gap_to_supervised.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).round(3).to_csv(args.out_dir / "gap_to_supervised.csv", index=False)
    print(f"wrote {out}")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
