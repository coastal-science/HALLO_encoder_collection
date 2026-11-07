"""SRKW-vs-rest precision / recall of SSAMBA-tiny C (SSL on all unlabelled audio, then fine-tuned) against the clean
EffNet-B1 fine-tune reference, each scored two ways: the fine-tuned classifier head, and the linear probe the
pipeline fits on that fold's fine-tuned checkpoint's embeddings (embeddings_fold<n> child run). Same test clips,
folds and scoring as compare_pretraining_results.py.

Writes to <out-dir>:
  ssamba_vs_effnet_pr_curves.png         test PR curve of every fold per site (row per head), with each fold's
                                         argmax and val-tuned operating points
  ssamba_vs_effnet_precision_recall.png  per-fold precision and recall per site (row per decision rule)
  ssamba_vs_effnet_summary.md            mean ± std over folds
  probe_per_species_ssl_vs_sl.png        linear probe only: per-species precision / recall boxplots per site

    python scripts/adaptation/plot_ssamba_vs_effnet.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from compare_pretraining_results import (COMPARISON_RUNS, N_FOLDS, SITES, TRACKING_URI, markdown, restrict_to_folds,
                                         scores, site_of)
from plot_exp3_comparison import srkw_threshold

REPO = Path(__file__).resolve().parents[2]
MODELS = {  # method -> (label, colour); reference palette slots 1 and 2
    "effnet_b1_ft": ("EffNet-B1", "#2a78d6"),
    "ssamba_C_ssl_all": ("SSAMBA-tiny C (SSL all)", "#eb6834"),
}
HEADS = {  # head -> (label, artifact of fold n, run it lives on, line style, filled marker)
    "finetune": ("fine-tuned head", "fold{fold}_predictions.csv", "parent", "-", True),
    "probe": ("linear probe on fine-tuned embeddings", "fold{fold}_linear_probe_predictions.csv", "embeddings_fold{fold}",
              "--", False),
}
RULES = {"argmax": ("argmax", "o"), "val_tuned": ("val-tuned threshold", "^")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3dd"


def predictions() -> pd.DataFrame:
    """val + test predictions of both heads, newest FINISHED pretraining-comparison run of each MODELS method / site."""
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("pretraining-comparison")
    runs = client.search_runs([exp.experiment_id], "attributes.status = 'FINISHED'", order_by=["attributes.start_time DESC"])
    frames, seen = [], set()
    for run in runs:
        prefix = next((p for p, m in COMPARISON_RUNS.items() if m in MODELS and run.info.run_name.startswith(p)), None)
        if prefix is None:
            continue
        method, site = COMPARISON_RUNS[prefix], site_of(run.info.run_name, prefix)
        if (method, site) in seen or site not in SITES:
            continue
        seen.add((method, site))
        children = {
            child.info.run_name: child.info.run_id
            for child in client.search_runs([exp.experiment_id], f"tags.mlflow.parentRunId = '{run.info.run_id}'")
        }
        for head, (_, artifact, run_of, _, _) in HEADS.items():
            for fold in range(N_FOLDS):
                run_id = run.info.run_id if run_of == "parent" else children.get(run_of.format(fold=fold))
                path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact.format(fold=fold))
                frames.append(pd.read_csv(path).assign(model=method, method=method, head=head, site=site, fold=fold))
    return pd.concat(frames)


def style(ax) -> None:
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_pr_curves(preds: pd.DataFrame, df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(len(HEADS), len(SITES), figsize=(15, 10), sharex=True, sharey=True)
    for row, (head, (head_label, _, _, linestyle, _)) in zip(axes, HEADS.items()):
        for ax, (site, (_, label)) in zip(row, SITES.items()):
            for mi, (method, (name, colour)) in enumerate(MODELS.items()):
                for fold, pred in preds[(preds.site == site) & (preds.method == method) & (preds["head"] == head)].groupby("fold"):
                    test = pred[pred.split == "test"]
                    y = (test.true_label == "SRKW").to_numpy()
                    precision, recall, _ = precision_recall_curve(y, test.prob_SRKW)
                    ax.plot(recall, precision, color=colour, linestyle=linestyle, linewidth=1.4, alpha=0.55)
                    threshold = srkw_threshold(pred[pred.split == "val"])
                    for rule, decision in {"argmax": test.pred_label == "SRKW", "val_tuned": test.prob_SRKW >= threshold}.items():
                        tp = (decision & y).sum()
                        ax.scatter(tp / y.sum(), tp / max(decision.sum(), 1), marker=RULES[rule][1], s=46, color=colour,
                                   edgecolor="white", linewidth=1.2, zorder=3)
                ap = df[(df.site == site) & (df.method == method) & (df["head"] == head) & (df.metric == "pr_auc")].value
                ax.text(0.02, 0.04 + 0.07 * mi, f"{name}: PR-AUC {ap.mean():.3f} ± {ap.std():.3f}",
                        transform=ax.transAxes, fontsize=9, color=INK)
            ax.set_title(f"{label} · {head_label}", loc="left", fontsize=11, color=INK)
            ax.set_xlim(0, 1.01)
            ax.set_ylim(0, 1.02)
            style(ax)
        row[0].set_ylabel("Precision", color=MUTED)
    for ax in axes[-1]:
        ax.set_xlabel("Recall", color=MUTED)
    handles = ([plt.Line2D([], [], color=c, linewidth=2) for _, c in MODELS.values()]
               + [plt.Line2D([], [], color=MUTED, linestyle=h[3], linewidth=1.5) for h in HEADS.values()]
               + [plt.Line2D([], [], marker=m, color=MUTED, linestyle="", markersize=7) for _, m in RULES.values()])
    labels = [n for n, _ in MODELS.values()] + [h[0] for h in HEADS.values()] + [f"{n} point" for n, _ in RULES.values()]
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.965), fontsize=9)
    fig.suptitle("SRKW vs rest: test PR curves, one line per fold (5 folds)", x=0.01, ha="left", y=0.99, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_precision_recall(df: pd.DataFrame, out: Path) -> None:
    series = [(method, head) for head in HEADS for method in MODELS]
    metrics = ("precision", "recall")
    fig, axes = plt.subplots(len(RULES), len(SITES), figsize=(15, 9))
    step = 0.19
    for row, (rule, (rule_label, _)) in zip(axes, RULES.items()):
        for ax, (site, (_, label)) in zip(row, SITES.items()):
            for xi, metric in enumerate(metrics):
                for si, (method, head) in enumerate(series):
                    colour, filled = MODELS[method][1], HEADS[head][4]
                    vals = df[(df.site == site) & (df.method == method) & (df["head"] == head) & (df.rule == rule)
                              & (df.metric == metric)].value
                    x = xi + (si - (len(series) - 1) / 2) * step
                    jitter = np.linspace(-0.03, 0.03, len(vals))
                    ax.scatter(x + jitter, vals, s=22, facecolor=colour if filled else "white", edgecolor=colour,
                               linewidth=1.0, alpha=0.75, zorder=3)
                    ax.errorbar(x, vals.mean(), yerr=vals.std(), fmt="none", ecolor=colour, elinewidth=1.5, capsize=4,
                                linestyle=HEADS[head][3], zorder=2)
                    ax.hlines(vals.mean(), x - 0.07, x + 0.07, color=colour, linewidth=2.5, zorder=4)
            ax.axvline(0.5, color=GRID, linewidth=1)
            ax.set_xticks(range(len(metrics)), metrics, fontsize=10)
            ax.set_xlim(-0.5, len(metrics) - 0.5)
            ax.set_title(f"{label} · {rule_label}", loc="left", fontsize=11, color=INK)
            style(ax)
            ax.grid(axis="x", visible=False)
        row[0].set_ylabel("SRKW test score", color=MUTED)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, markeredgecolor=MODELS[m][1],
                          markerfacecolor=MODELS[m][1] if HEADS[h][4] else "white") for m, h in series]
    fig.legend(handles, [f"{MODELS[m][0]}, {HEADS[h][0]}" for m, h in series], loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 0.965), fontsize=9)
    fig.suptitle("SRKW vs rest: precision and recall per fold (dots), mean ± std (bar); y-axes differ per panel",
                 x=0.01, ha="left", y=0.99, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_probe_per_class(preds: pd.DataFrame, out: Path) -> None:
    """Linear probe only: per-class test precision / recall of the 4-class argmax, one box of 5 folds per model."""
    test = preds[(preds["head"] == "probe") & (preds.split == "test")]
    rows = []
    for (method, site, fold), pred in test.groupby(["method", "site", "fold"]):
        for species in sorted(pred.true_label.unique()):
            y, p = pred.true_label == species, pred.pred_label == species
            rows += [dict(method=method, site=site, fold=fold, species=species, n=y.sum(),
                          precision=(y & p).sum() / max(p.sum(), 1), recall=(y & p).sum() / y.sum())]
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, len(SITES), figsize=(15, 9), sharey=True)
    width = 0.32
    for row, metric in zip(axes, ("precision", "recall")):
        for ax, (site, (_, label)) in zip(row, SITES.items()):
            s = df[df.site == site]
            species = sorted(s.species.unique())
            for mi, (method, (_, colour)) in enumerate(MODELS.items()):
                for xi, sp in enumerate(species):
                    vals = s[(s.method == method) & (s.species == sp)][metric]
                    x = xi + (mi - 0.5) * width * 1.15
                    ax.boxplot(vals, positions=[x], widths=width, patch_artist=True, showfliers=False,
                               boxprops=dict(facecolor=colour + "40", edgecolor=colour, linewidth=1.5),
                               medianprops=dict(color=colour, linewidth=2), whiskerprops=dict(color=colour, linewidth=1.2),
                               capprops=dict(color=colour, linewidth=1.2))
                    ax.scatter(x + np.linspace(-0.05, 0.05, len(vals)), vals, s=14, color=colour, edgecolor="white",
                               linewidth=0.8, zorder=3)
            n = s.groupby("species").n.first()
            ax.set_xticks(range(len(species)), [f"{sp}\nn={n[sp]}" for sp in species], fontsize=9)
            ax.set_xlim(-0.6, len(species) - 0.4)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{label} · {metric}", loc="left", fontsize=11, color=INK)
            style(ax)
            ax.grid(axis="x", visible=False)
        row[0].set_ylabel(metric.capitalize(), color=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c + "40", edgecolor=c, linewidth=1.5) for _, c in MODELS.values()]
    fig.legend(handles, ["EffNet-B1 (supervised pretraining)", "SSAMBA-tiny C (SSL pretraining)"], loc="upper center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965), fontsize=10)
    fig.suptitle("Linear probe on fine-tuned embeddings: per-species test precision and recall, argmax, 5 folds "
                 "(n = test clips per fold)", x=0.01, ha="left", y=0.99, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def write_summary(df: pd.DataFrame, out: Path) -> None:
    summary = df.groupby(["site", "method", "head", "rule", "metric"]).value.agg(["mean", "std"]).reset_index()
    summary["cell"] = summary.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    summary["col"] = np.where(summary.rule == "any", summary.metric, summary.rule + " " + summary.metric)
    cols = ["pr_auc"] + [f"{r} {m}" for r in RULES for m in ("precision", "recall", "f1")]
    lines = ["# SRKW vs rest, SSAMBA-tiny C vs EffNet-B1: fine-tuned head and linear probe, mean ± std over folds", ""]
    for site, (_, label) in SITES.items():
        table = summary[summary.site == site].pivot_table(index=["method", "head"], columns="col", values="cell",
                                                          aggfunc="first").reset_index()
        table = table.sort_values(["head", "method"], key=lambda c: c.map({k: i for i, k in enumerate(HEADS | MODELS)}))
        lines += [f"## {label}", "", markdown(table[["method", "head"] + cols]), ""]
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPO / "scripts/adaptation/results/pretraining_comparison")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)

    preds = restrict_to_folds(predictions())
    rows = []
    for (method, head, site, fold), pred in preds.groupby(["method", "head", "site", "fold"]):
        rows += [dict(method=method, head=head, site=site, fold=fold, **r) for r in scores(pred)]
    df = pd.DataFrame(rows)

    write_summary(df, args.out_dir / "ssamba_vs_effnet_summary.md")
    plot_precision_recall(df, args.out_dir / "ssamba_vs_effnet_precision_recall.png")
    plot_pr_curves(preds, df, args.out_dir / "ssamba_vs_effnet_pr_curves.png")
    plot_probe_per_class(preds, args.out_dir / "probe_per_species_ssl_vs_sl.png")


if __name__ == "__main__":
    main()
