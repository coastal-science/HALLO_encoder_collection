"""Figure for compare_file_level_from_clips.py: Strait of Georgia, SRKW vs rest at file level.

Left: mean file-level PR curve over the 5 folds (file score = max over its clips), with each model's deployed file
decision (BirdNET-style vote of its own clip decisions) as a ringed dot at its mean precision / recall.
Right: errors of each model's deployed file decision: SRKW files missed and negative files called SRKW, rows ordered
by file PR-AUC. (Val-tuned F1 is not shown: with 73 % SRKW files the tuned threshold collapses to "every file SRKW".)

    python scripts/adaptation/plot_file_level_from_clips.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score, precision_recall_curve

from plot_ssamba_vs_effnet import INK, MUTED, style

RESULTS = Path(__file__).resolve().parent / "results/file_level_from_clips"
SITE = "strait_of_georgia"
# colours validated all-pairs (validate_palette.js, light): worst CVD dE 10.0, normal-vision dE 22.1
METHODS = {  # method -> (label, colour, linestyle, site-trained)
    "perch_v2_supervised_source": ("Perch v2 fine-tuned (G)", "#b8336a", "-", True),
    "perch_v2_probe": ("Perch v2 frozen (probe)", "#b8336a", (0, (5, 2)), True),
    "effnet_b1_ft": ("EffNet-B1 fine-tuned", "#3f51b5", "-", True),
    "zeroshot": ("EffNet-B1 zero-shot (exp_2)", "#898781", (0, (4, 3)), False),
    "hallo_v1": ("HALLO encoder v1", "#00a3c4", "-", False),
    "birdnet": ("BirdNET", "#eda100", "-", False),
    "sequence": ("Sequence KW detector", "#7a7a00", "-", False),
    "cascade": ("Deployed cascade (seq ∧ BirdNET)", INK, (0, (1, 1.5)), False),
}
GRID = np.linspace(0, 1, 201)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    files = pd.read_csv(args.in_dir / "files.csv")
    files = files[(files.site == SITE) & (files.split == "test")]
    files = files.assign(y=files.labels.str.split("+").map(lambda s: "SRKW" in s))
    metrics = pd.read_csv(args.in_dir / "metrics_per_fold.csv")
    metrics = metrics[(metrics.site == SITE) & (metrics.task == "SRKW")]
    mean = metrics.groupby(["method", "rule", "metric"]).value.mean()

    order = sorted(METHODS, key=lambda m: -mean[(m, "any", "pr_auc")])
    n_files, n_pos = len(files[files.method == order[0]].query("fold == 0")), int(files.query("fold == 0").drop_duplicates("LocalPath").y.sum())

    fig = plt.figure(figsize=(15, 5.8), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[1, 1.15], wspace=0.02)
    ax = left.subplots()
    bx, cx_ = right.subplots(1, 2, sharey=True, gridspec_kw=dict(wspace=0.05))
    handles = []
    for m in order:
        label, colour, ls, _ = METHODS[m]
        curves = []
        for _, f in files[files.method == m].groupby("fold"):
            precision, recall, _ = precision_recall_curve(f.y, f.srkw_score)
            curves.append(np.interp(GRID, recall[::-1], np.maximum.accumulate(precision)[::-1]))
        ax.plot(GRID, np.mean(curves, axis=0), color=colour, linestyle=ls, linewidth=2, zorder=3)
        p, r = mean[(m, "vote", "precision")], mean[(m, "vote", "recall")]
        ax.scatter(r, p, s=70, color=colour, edgecolor="white", linewidth=2, zorder=5)
        handles.append(Line2D([], [], color=colour, linestyle=ls, linewidth=2,
                              label=f"{label}  {mean[(m, 'any', 'pr_auc')]:.3f}"))
    handles.append(Line2D([], [], color=MUTED, marker="o", linestyle="", markersize=8, markeredgecolor="white",
                          label="deployed file decision (vote)"))
    ax.legend(handles=handles, title="file PR-AUC (mean of 5 folds)", title_fontsize=8.5, fontsize=8.5, loc="lower left",
              frameon=True, facecolor="white", edgecolor="none", framealpha=0.92, labelcolor=INK, handlelength=2.6)
    chance = n_pos / n_files
    ax.axhline(chance, color=MUTED, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
    ax.text(0.99, chance - 0.015, f"chance (every file SRKW) {chance:.2f}", ha="right", va="top", fontsize=8, color=MUTED)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (SRKW files found)", color=MUTED)
    ax.set_ylabel("Precision", color=MUTED)
    ax.set_title("File-level PR curves", loc="left", fontsize=11, color=INK)
    style(ax)

    n_neg = n_files - n_pos
    labels = [METHODS[m][0] + ("" if METHODS[m][3] else " †") for m in order]
    for cx, (key, total, title) in zip((bx, cx_), [("missed", n_pos, f"SRKW files missed (of {n_pos})"),
                                                   ("fp", n_neg, f"Negative files called SRKW (of {n_neg})")]):
        for yi, m in enumerate(order):
            colour = METHODS[m][1]
            v = (1 - mean[(m, "vote", "recall")]) * n_pos if key == "missed" else mean[(m, "vote", "fp_files")]
            cx.barh(yi, v, height=0.62, color=colour, edgecolor="white", linewidth=2, zorder=3)
            cx.annotate(f"{v:.0f}", (v, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
        cx.set_xlim(0, total)
        cx.set_title(title, loc="left", fontsize=10.5, color=INK)
        style(cx)
        cx.grid(axis="y", visible=False)
    bx.set_yticks(range(len(order)), labels, fontsize=9, color=INK)
    bx.invert_yaxis()
    cx_.tick_params(axis="y", left=False, labelleft=False)
    bx.set_xlabel("files", color=MUTED)
    cx_.set_xlabel("files", color=MUTED)

    fig.suptitle(f"Strait of Georgia, SRKW vs rest per file: {n_files} test files ({n_pos} SRKW), file score = max over its "
                 "annotated clips\nDots and bars = each model's deployed file decision (vote).  "
                 "† not trained on this site.  Site-trained models: mean of 5 folds", x=0.01, ha="left", fontsize=11.5, color=INK)
    out = args.in_dir / "file_level_srkw_strait_of_georgia.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
