"""Experiment E: SRKW-vs-rest score against the fraction of target-site labels each arm is fine-tuned on.

10 / 25 / 50 % = the label_budget_frac<N>_* runs (configs/pretraining_comparison/E_budget_*, splits from
scripts/adaptation/make_label_budget_splits.py: each fold's train / val files subsampled, test untouched);
100 % = the same arm's pretraining-comparison run. Arms: SSAMBA-tiny A (supervised source), B (SSL on A's source
clips; only its 100 % point until it has E tiers), C5 (SSL on all audio, 5 ep), D (scratch), and the Perch v2
probe. Every tier is scored on exactly the 100 % runs' test clips, with compare_pretraining_results.py's rules
(argmax; val_tuned = threshold maximising SRKW F1 on that tier's own, subsampled, val split). The exp_2 effnet_b1
zero-shot model (no target labels) is drawn as a reference line.

Writes <out-dir>/budget_metrics_per_fold.csv, budget_summary.csv / budget_summary.md (per site: mean +- std over
folds per arm and tier, the labelled clips per tier, and the smallest tier at which each arm reaches A's 100 % F1)
and budget_<rule>.png (rows = F1 / precision / recall, columns = sites).

    python scripts/adaptation/compare_label_budget.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from compare_pretraining_results import (COMPARISON_RUNS, METRICS, REPO, SITES, SPLITS, TRACKING_URI, comparison_predictions,
                                         markdown, restrict_to_folds, scores)

FRACS = [10, 25, 50]
BUDGET_SPLITS = REPO / "data_raw/DORI/20261022_190549/splits_label_budget_frac{frac}_ft_{dep}.csv"
ANNOTATIONS = REPO / "data_raw/DORI/20261022_190549/annotations_bgcap10000_ft_{dep}.csv"
ARMS = {  # method -> (label, colour); fixed per arm, validated all-pairs (worst CVD dE 10.0, normal-vision dE 22.1)
    "ssamba_A_supervised": ("A supervised", "#3f51b5"),
    "ssamba_B_ssl_source": ("B SSL source clips", "#00a3c4"),
    "ssamba_C5_ssl_all_5ep": ("C5 SSL all audio", "#7a7a00"),
    "ssamba_D_scratch": ("D scratch", "#eda100"),
    "perch_v2_probe": ("Perch v2 probe", "#b8336a"),
    # G / H fine-tuned Perch: the Perch colour, told apart from the probe by marker + line style (no 6th hue passes all-pairs CVD)
    "perch_v2_supervised_source": ("Perch v2 G source + ft", "#b8336a"),
    "perch_v2_direct_ft": ("Perch v2 H direct ft", "#b8336a"),
    # F frozen-encoder probes: same colour and marker as the encoder's fine-tuned arm, dashed line (LINESTYLES)
    "probe_A_supervised": ("A frozen probe", "#3f51b5"),
    "probe_B_ssl_source": ("B frozen probe", "#00a3c4"),
    "probe_C5_ssl_all_5ep": ("C5 frozen probe", "#7a7a00"),
}
MARKERS = {"ssamba_A_supervised": "o", "ssamba_B_ssl_source": "s", "ssamba_C5_ssl_all_5ep": "^", "ssamba_D_scratch": "D",
           "perch_v2_probe": "v", "probe_A_supervised": "o", "probe_B_ssl_source": "s",
           "probe_C5_ssl_all_5ep": "^", "perch_v2_supervised_source": "P", "perch_v2_direct_ft": "X"}  # second cue besides colour
LINESTYLES = {"perch_v2_probe": (0, (5, 2)), "perch_v2_direct_ft": (0, (1, 1.5)), "probe_A_supervised": (0, (5, 2)), "probe_B_ssl_source": (0, (5, 2)), "probe_C5_ssl_all_5ep": (0, (5, 2))}
REFERENCE = "ssamba_A_supervised"  # "labels to match": the tier at which an arm reaches this arm's 100 % F1


def budget_prefixes() -> dict[str, str]:
    """run name prefix -> '<method>@<frac>' for every arm's 100 % run and its label_budget_frac<N>_ tiers."""
    prefixes = {}
    for prefix, method in COMPARISON_RUNS.items():
        if method not in ARMS:
            continue
        prefixes[prefix] = f"{method}@100"
        for frac in FRACS:
            prefixes[f"label_budget_frac{frac}_{prefix.removeprefix('pretrain_cmp_')}"] = f"{method}@{frac}"
    return prefixes


def labelled_clips() -> pd.DataFrame:
    """mean over folds of the train clips (all / SRKW) and train files each tier fine-tunes on, per site."""
    rows = []
    for site, (dep, _) in SITES.items():
        ann = pd.read_csv(str(ANNOTATIONS).format(dep=dep), usecols=["uid", "Labels", "LocalPath"], low_memory=False)
        for frac in FRACS + [100]:
            path = str(SPLITS).format(dep=dep) if frac == 100 else str(BUDGET_SPLITS).format(frac=frac, dep=dep)
            splits = pd.read_csv(path).merge(ann, on="uid")
            folds = [c for c in splits.columns if c.startswith("fold_")]
            train = [splits[splits[f] == "train"] for f in folds]
            rows.append(dict(site=site, frac=frac, train_clips=np.mean([len(t) for t in train]),
                             train_srkw=np.mean([(t.Labels == "SRKW").sum() for t in train]),
                             train_files=np.mean([t.LocalPath.nunique() for t in train])))
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, zeroshot: pd.DataFrame, rule: str, out: Path) -> None:
    s = summary[summary.rule == rule]
    fig, axes = plt.subplots(len(METRICS), len(SITES), figsize=(4.2 * len(SITES), 3.2 * len(METRICS)), sharex=True, sharey="row")
    for row, (metric, (metric_label, _)) in enumerate(METRICS.items()):
        for col, (site, (_, site_label)) in enumerate(SITES.items()):
            ax = axes[row, col]
            ref = zeroshot[(zeroshot.site == site) & (zeroshot.rule == rule) & (zeroshot.metric == metric)]["mean"]
            if not ref.empty:
                ax.axhline(ref.iloc[0], color="#898781", linewidth=1, linestyle=(0, (4, 3)), zorder=1)
                if row == 0 and col == 0:
                    ax.text(10, ref.iloc[0], " zero-shot effnet", color="#5f5e5a", fontsize=7, va="bottom")
            for method, (label, colour) in ARMS.items():
                m = s[(s.site == site) & (s.method == method) & (s.metric == metric)]
                if m.empty:
                    continue
                m = m.set_index("frac").reindex(FRACS + [100]).reset_index()  # NaN at unrun tiers breaks the line there
                ax.fill_between(m.frac, m["mean"] - m["std"], m["mean"] + m["std"], color=colour, alpha=0.12, linewidth=0)
                ax.plot(m.frac, m["mean"], color=colour, linewidth=2, linestyle=LINESTYLES.get(method, "-"), marker=MARKERS[method], markersize=6,
                        markeredgecolor="white", markeredgewidth=1, label=label, zorder=3)
            ax.set_xscale("log")
            ax.set_xticks(FRACS + [100], [f"{f}%" for f in FRACS + [100]])
            ax.minorticks_off()
            ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(site_label, loc="left", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"SRKW {metric_label}")
            if row == len(METRICS) - 1:
                ax.set_xlabel("target-site labelled files used")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(f"Label efficiency: SRKW vs rest on the 100 % runs' test clips, mean ± std over 5 folds, {rule} decision",
                 x=0.01, ha="left", y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adabn-predictions", type=Path,
                        default=REPO / "scripts/adaptation/results/exp3_splits/adabn_exp3_predictions.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO / "scripts/adaptation/results/pretraining_comparison")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)

    missing: list[str] = []
    preds = restrict_to_folds(comparison_predictions(missing, budget_prefixes()))
    rows = []
    for (site, fold, tagged), pred in preds.groupby(["site", "fold", "method"]):
        method, frac = tagged.split("@")
        rows += [dict(site=site, fold=fold, method=method, frac=int(frac), **r) for r in scores(pred)]
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "budget_metrics_per_fold.csv", index=False)
    summary = df.groupby(["site", "method", "frac", "rule", "metric"]).value.agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(args.out_dir / "budget_summary.csv", index=False)

    adabn = pd.read_csv(args.adabn_predictions).rename(columns={"variant": "method"})
    adabn = restrict_to_folds(adabn[(adabn.method == "zeroshot") & (adabn.model == "effnet_b1")])
    zeroshot = pd.DataFrame([dict(site=site, fold=fold, **r) for (site, fold), p in adabn.groupby(["site", "fold"]) for r in scores(p)])
    zeroshot = zeroshot.groupby(["site", "rule", "metric"]).value.mean().rename("mean").reset_index()

    # every tier must be scored on the 100 % runs' test clips
    expected = {s: int((pd.read_csv(str(SPLITS).format(dep=d)).fold_0 == "test").sum()) for s, (d, _) in SITES.items()}
    short = df.groupby(["site", "method", "frac"]).n_test.min().reset_index()
    short = short[short.n_test < short.site.map(expected)]

    clips = labelled_clips()
    lines = ["# Label budget (E): SRKW vs rest, mean ± std over folds", "",
             "Columns = share of the target site's labelled train / val files kept. val_tuned uses each tier's own val split.", ""]
    for site, (_, label) in SITES.items():
        s = summary[summary.site == site]
        if s.empty:
            continue
        c = clips[clips.site == site]
        lines += [f"## {label} ({expected[site]} test clips)", "", "Labelled train set per fold (mean):", "",
                  markdown(pd.DataFrame({"": ["clips", "SRKW clips", "files"]} | {
                      f"{r.frac}%": [f"{r.train_clips:.0f}", f"{r.train_srkw:.0f}", f"{r.train_files:.0f}"] for r in c.itertuples()})), ""]
        for rule, metric in [("val_tuned", "f1"), ("val_tuned", "precision"), ("val_tuned", "recall"), ("argmax", "f1"), ("any", "pr_auc")]:
            m = s[(s.rule == rule) & (s.metric == metric)].copy()
            m["cell"] = m.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}" if r["count"] > 1 else f"{r['mean']:.3f}", axis=1)
            table = m.pivot_table(index="method", columns="frac", values="cell", aggfunc="first")
            table = table.reindex([a for a in ARMS if a in table.index])
            table.columns = [f"{f}%" for f in table.columns]
            table.insert(0, "arm", [ARMS[a][0] for a in table.index])
            zs = zeroshot[(zeroshot.site == site) & (zeroshot.rule == rule) & (zeroshot.metric == metric)]["mean"]
            note = f" (zero-shot effnet_b1: {zs.iloc[0]:.3f})" if not zs.empty else ""
            lines += [f"### {'' if rule == 'any' else rule + ' '}{metric}{note}", "", markdown(table), ""]
        # smallest tier whose mean val_tuned F1 reaches the reference arm's 100 % mean
        f1 = s[(s.rule == "val_tuned") & (s.metric == "f1")]
        target = f1[(f1.method == REFERENCE) & (f1.frac == 100)]["mean"]
        if not target.empty:
            reach = []
            for method in [a for a in ARMS if a in set(f1.method)]:
                hit = f1[(f1.method == method) & (f1["mean"] >= target.iloc[0])].frac
                reach.append(dict(arm=ARMS[method][0], tier=f"{hit.min()}%" if not hit.empty else "not reached",
                                  tiers_run=", ".join(f"{f}%" for f in sorted(f1[f1.method == method].frac))))
            lines += [f"### Smallest tier reaching {ARMS[REFERENCE][0]} at 100 % (val_tuned F1 {target.iloc[0]:.3f})", "",
                      markdown(pd.DataFrame(reach)), ""]
    if not short.empty:
        lines += ["## Tiers scored on fewer test clips than expected", "", markdown(short), ""]
    if missing:
        lines += ["## Not included (missing / unfinished)", ""] + [f"- {m}" for m in missing] + [""]
    (args.out_dir / "budget_summary.md").write_text("\n".join(lines))
    print(f"wrote {args.out_dir / 'budget_summary.md'}")
    for rule in ("argmax", "val_tuned"):
        plot(summary, zeroshot, rule, args.out_dir / f"budget_{rule}.png")


if __name__ == "__main__":
    main()
