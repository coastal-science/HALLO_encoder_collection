"""File-level SRKW-vs-rest PR curves, one panel per deployment (from compare_file_level_from_clips.py's files.csv).

File score = max SRKW score over the file's annotated clips. Site-trained models: mean curve of the 5 folds (the test
files are the same in every fold). Dots = each model's deployed-style file decision (vote of its clip decisions), mean
precision / recall over folds. The hallo-tuned-detectors models (HALLO v1, BirdNET, sequence, cascade) are only at
Strait of Georgia: LimeKiln and Tekteksen files are in their training data.

    python scripts/adaptation/plot_file_level_pr_by_site.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve

from plot_file_level_from_clips import GRID, METHODS, RESULTS
from plot_ssamba_vs_effnet import INK, MUTED, style

SITES = {"strait_of_georgia": "Strait of Georgia", "limekiln": "LimeKiln", "tekteksen": "Tekteksen"}


def main() -> None:
    files = pd.read_csv(RESULTS / "files.csv")
    files = files[files.split == "test"].assign(y=lambda f: f.labels.str.split("+").map(lambda s: "SRKW" in s))
    metrics = pd.read_csv(RESULTS / "metrics_per_fold.csv")
    mean = metrics[metrics.task == "SRKW"].groupby(["site", "method", "rule", "metric"]).value.mean()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), layout="constrained")
    for ax, (site, site_label) in zip(axes, SITES.items()):
        f_site = files[files.site == site]
        present = [m for m in METHODS if m in set(f_site.method)]
        order = sorted(present, key=lambda m: -mean[(site, m, "any", "pr_auc")])
        one = f_site[(f_site.method == order[0]) & (f_site.fold == 0)]
        n_files, n_pos = len(one), int(one.y.sum())
        handles = []
        for m in order:
            label, colour, ls, trained = METHODS[m]
            curves = []
            for _, f in f_site[f_site.method == m].groupby("fold"):
                precision, recall, _ = precision_recall_curve(f.y, f.srkw_score)
                curves.append(np.interp(GRID, recall[::-1], np.maximum.accumulate(precision)[::-1]))
            ax.plot(GRID, np.mean(curves, axis=0), color=colour, linestyle=ls, linewidth=2, zorder=3)
            ax.scatter(mean[(site, m, "vote", "recall")], mean[(site, m, "vote", "precision")], s=70, color=colour,
                       edgecolor="white", linewidth=2, zorder=5)
            handles.append(Line2D([], [], color=colour, linestyle=ls, linewidth=2,
                                  label=f"{label}{'' if trained else ' †'}  {mean[(site, m, 'any', 'pr_auc')]:.3f}"))
        handles.append(Line2D([], [], color=MUTED, marker="o", linestyle="", markersize=8, markeredgecolor="white",
                              label="deployed-style file decision (vote)"))
        ax.legend(handles=handles, title="file PR-AUC", title_fontsize=8.5, fontsize=8.3, loc="lower left", frameon=True,
                  facecolor="white", edgecolor="none", framealpha=0.92, labelcolor=INK, handlelength=2.6)
        chance = n_pos / n_files
        ax.axhline(chance, color=MUTED, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
        ax.text(0.01, chance - 0.012, f"chance {chance:.2f}", ha="left", va="top", fontsize=8, color=MUTED)
        ax.set(xlim=(0, 1.01), ylim=(0, 1.04))
        ax.set_xlabel("Recall (SRKW files found)", color=MUTED)
        ax.set_ylabel("Precision", color=MUTED)
        ax.set_title(f"{site_label}: {n_files} test files ({n_pos} SRKW, {n_files - n_pos} other)", loc="left",
                     fontsize=11, color=INK)
        style(ax)
    fig.suptitle("File-level PR curves, SRKW vs rest, per deployment (file score = max over its annotated clips; "
                 "site-trained models = mean of 5 folds)\n† not trained on the site. hallo-tuned-detectors models only at "
                 "Strait of Georgia (the other two sites are in their training data). Tekteksen has 1 negative file, so "
                 "its curves can't separate models.", x=0.01, ha="left", fontsize=11, color=INK)
    out = RESULTS / "file_level_pr_by_site.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
