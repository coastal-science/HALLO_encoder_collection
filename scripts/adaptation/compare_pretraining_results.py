"""SRKW-vs-rest results of every method in configs/pretraining_comparison, plus the AdaBN / zero-shot runs and
exp_3's original fine-tune, all scored on exactly the same clips: each fold's val and test rows of the
comparison's folds (data_raw/DORI/<run>/splits_pretrain_cmp_ft_<site>.csv, from
scripts/DORI/export_config_splits.py), i.e. exp_3's folds minus the NaN-window rows. Predictions outside them
(e.g. the 14 NaN-window test clips per fold in exp_3 / AdaBN's Strait of Georgia test set) are dropped first.

Methods, by how many target-site labels they use (* = uses target-site labels):
  zero-shot, AdaBN (adabn_exp3.py; the two balanced variants pick their adaptation clips with train labels *),
  Perch v2 linear probe *, and the fully fine-tuned arms * -- effnet_b1 reference, SSAMBA-tiny A / B / C (C from
  exp_1's early-stopped epoch-1 checkpoint; C 4 ep from the C_ssl_all_4ep contingency, when it has run; C 5 ep from
  C_ssl_all_5ep, exp_1's SSL continued to 5 epochs; D = SSAMBA-tiny from random init, no pretraining), and
  exp_3's original effnet_b1 fine-tune (trained with array-channel copies of its test calls, so optimistic on
  Strait of Georgia). A run that is missing or unfinished is skipped and listed.

Decision rules, as plot_exp3_comparison.py: argmax (SRKW iff the 4-class argmax) and val_tuned (prob_SRKW >= the
threshold maximising SRKW-vs-rest F1 on that fold's val split). PR-AUC is threshold-free.

Writes <out-dir>/metrics_per_fold.csv, summary.csv / summary.md (mean +- std over folds) and one figure per rule.

    python scripts/adaptation/compare_pretraining_results.py
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from plot_exp3_comparison import exp3_predictions, srkw_threshold

REPO = Path(__file__).resolve().parents[2]
TRACKING_URI = "http://localhost:5000"
SPLITS = REPO / "data_raw/DORI/20261022_190549/splits_pretrain_cmp_ft_{dep}.csv"
SITES = {  # site -> (annotation / splits slug, figure label)
    "strait_of_georgia": ("dori_venus_strait_of_georgia", "Strait of Georgia"),
    "limekiln": ("limekiln", "LimeKiln"),
    "tekteksen": ("tekteksen", "Tekteksen"),
}
N_FOLDS = 5
METHODS = {  # method -> figure label; in increasing order of target labels used
    "zeroshot": "zero-shot",
    "adabn_train": "AdaBN",
    "adabn_balanced": "AdaBN\nbalanced*",
    "adabn_balanced_srkw_rest": "AdaBN\n50/50*",
    "perch_v2_probe": "Perch v2\nprobe*",
    "probe_A_supervised": "SSAMBA A\nfrozen probe*",
    "probe_B_ssl_source": "SSAMBA B\nfrozen probe*",
    "probe_C5_ssl_all_5ep": "SSAMBA C5\nfrozen probe*",
    "effnet_b1_ft": "EffNet-B1\nfine-tune*",
    "ssamba_A_supervised": "SSAMBA A\nsupervised*",
    "ssamba_B_ssl_source": "SSAMBA B\nSSL source*",
    "ssamba_C_ssl_all": "SSAMBA C\nSSL all, 1 ep*",
    "ssamba_C4_ssl_all_4ep": "SSAMBA C\nSSL all, 4 ep*",
    "ssamba_C5_ssl_all_5ep": "SSAMBA C\nSSL all, 5 ep*",
    "ssamba_D_scratch": "SSAMBA D\nscratch*",
    "perch_v2_supervised_source": "Perch v2\nsource + ft*",
    "perch_v2_direct_ft": "Perch v2\ndirect ft*",
    "exp3_effnet_b1_ft": "exp_3 EffNet\n(leaky)*",
}
COMPARISON_RUNS = {  # pretraining-comparison run name prefix -> method
    "pretrain_cmp_effnet_b1_ft_": "effnet_b1_ft",
    "pretrain_cmp_ssamba_tiny_supervised_source_ft_": "ssamba_A_supervised",
    "pretrain_cmp_ssamba_tiny_ssl_source_ft_": "ssamba_B_ssl_source",
    "pretrain_cmp_ssamba_tiny_ssl_all_ft_": "ssamba_C_ssl_all",
    "pretrain_cmp_ssamba_tiny_ssl_all_4ep_ft_": "ssamba_C4_ssl_all_4ep",
    "pretrain_cmp_ssamba_tiny_ssl_all_5ep_ft_": "ssamba_C5_ssl_all_5ep",
    "pretrain_cmp_ssamba_tiny_scratch_ft_": "ssamba_D_scratch",
    "pretrain_cmp_perch_v2_probe_ft_": "perch_v2_probe",
    # G / H: fine-tuned Perch v2 backbone (G: Google init -> A's supervised source pretrain -> fine-tune; H: Google init -> fine-tune)
    "pretrain_cmp_perch_v2_supervised_source_ft_": "perch_v2_supervised_source",
    "pretrain_cmp_perch_v2_direct_ft_": "perch_v2_direct_ft",
    # F: linear probe on the FROZEN pretrained encoders (configs/pretraining_comparison/F_frozen_probe_*), Perch's protocol
    "pretrain_cmp_frozen_probe_ssamba_tiny_supervised_source_ft_": "probe_A_supervised",
    "pretrain_cmp_frozen_probe_ssamba_tiny_ssl_source_ft_": "probe_B_ssl_source",
    "pretrain_cmp_frozen_probe_ssamba_tiny_ssl_all_5ep_ft_": "probe_C5_ssl_all_5ep",
}
METRICS = {"precision": ("Precision", "#2a78d6"), "recall": ("Recall", "#eb6834"), "f1": ("F1", "#1baf7a")}


def site_of(run_name: str, prefix: str) -> str:
    slug = re.sub(r"_5fold$", "", run_name.removeprefix(prefix)).removeprefix("dori_venus_")
    return slug


def comparison_predictions(missing: list[str], prefixes: dict[str, str] = COMPARISON_RUNS) -> pd.DataFrame:
    """val + test predictions of the newest FINISHED pretraining-comparison run of each method / site: the
    fine-tune's fold<n>_predictions.csv, or for a linear-probe arm (Perch, F frozen probes) its embeddings_fold<n> child's probe predictions.
    prefixes maps run name prefix -> method (compare_label_budget.py passes the label_budget_* runs too)."""
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("pretraining-comparison")
    runs = client.search_runs([exp.experiment_id], "attributes.status = 'FINISHED'", order_by=["attributes.start_time DESC"],
                              max_results=5000) if exp else []
    frames, seen = [], set()
    for run in runs:
        prefix = next((p for p in prefixes if run.info.run_name.startswith(p)), None)
        if prefix is None:
            continue
        method, site = prefixes[prefix], site_of(run.info.run_name, prefix)
        if (method, site) in seen or site not in SITES:
            continue
        seen.add((method, site))
        children = {
            child.info.run_name: child.info.run_id
            for child in client.search_runs([exp.experiment_id], f"tags.mlflow.parentRunId = '{run.info.run_id}'")
        }
        for fold in range(N_FOLDS):
            run_id, artifact = (
                (children.get(f"embeddings_fold{fold}"), f"fold{fold}_linear_probe_predictions.csv")
                if "probe" in method else (run.info.run_id, f"fold{fold}_predictions.csv")
            )
            try:
                frame = pd.read_csv(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact))
            except Exception as e:  # e.g. a fold whose artifact never got logged
                missing.append(f"{method} / {site} / fold {fold}: {type(e).__name__}")
                continue
            frames.append(frame.assign(model=method, site=site, fold=fold, method=method))
    for method in prefixes.values():
        missing += [f"{method} / {site}: no FINISHED run" for site in SITES if (method, site) not in seen]
    return pd.concat(frames) if frames else pd.DataFrame()


def restrict_to_folds(preds: pd.DataFrame) -> pd.DataFrame:
    """Only each fold's comparison val / test rows."""
    keep = []
    for site, (dep, _) in SITES.items():
        splits = pd.read_csv(str(SPLITS).format(dep=dep))
        for fold in range(N_FOLDS):
            rows = splits[["uid", f"fold_{fold}"]].dropna().rename(columns={f"fold_{fold}": "split"})
            keep.append(rows[rows.split.isin(["val", "test"])].assign(site=site, fold=fold))
    keep = pd.concat(keep)
    return preds.astype({"uid": np.int64}).merge(keep.astype({"uid": np.int64}), on=["uid", "split", "site", "fold"])


def markdown(table: pd.DataFrame) -> str:
    """table as a GitHub markdown table (pandas' to_markdown needs tabulate, which the venv lacks)."""
    rows = [list(map(str, table.columns))] + table.astype(str).values.tolist()
    return "\n".join(["| " + " | ".join(r) + " |" for r in rows[:1]] + ["|" + "---|" * len(table.columns)]
                     + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def scores(pred: pd.DataFrame) -> list[dict]:
    test = pred[pred.split == "test"]
    y = (test.true_label == "SRKW").to_numpy()
    threshold = srkw_threshold(pred[pred.split == "val"])
    rows = [dict(rule="any", metric="pr_auc", value=average_precision_score(y, test.prob_SRKW), n_test=len(test))]
    for rule, p in {"argmax": (test.pred_label == "SRKW").to_numpy(), "val_tuned": (test.prob_SRKW >= threshold).to_numpy()}.items():
        rows += [
            dict(rule=rule, metric=name, value=fn(y, p, zero_division=0), n_test=len(test))
            for name, fn in [("precision", precision_score), ("recall", recall_score), ("f1", f1_score)]
        ]
    return rows


def plot(df: pd.DataFrame, rule: str, out: Path) -> None:
    df = df[(df.rule == rule) & (df.model.isin(["effnet_b1"]) | ~df.method.str.startswith(("zeroshot", "adabn")))]
    methods = [m for m in METHODS if m in set(df.method)]
    fig, axes = plt.subplots(1, len(SITES), figsize=(4 + 2.1 * len(methods), 5), sharey=True)
    width = 0.25
    offsets = {m: (i - (len(METRICS) - 1) / 2) * width * 1.15 for i, m in enumerate(METRICS)}
    for ax, (site, (_, label)) in zip(axes, SITES.items()):
        for xi, method in enumerate(methods):
            for metric, (_, colour) in METRICS.items():
                vals = df[(df.site == site) & (df.method == method) & (df.metric == metric)].value
                if vals.empty:
                    continue
                x = xi + offsets[metric]
                ax.boxplot(vals, positions=[x], widths=width, patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=colour + "40", edgecolor=colour, linewidth=1.5),
                           medianprops=dict(color=colour, linewidth=2), whiskerprops=dict(color=colour, linewidth=1.2),
                           capprops=dict(color=colour, linewidth=1.2))
                ax.scatter([x] * len(vals), vals, s=14, color=colour, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_title(label, loc="left", fontsize=11)
        ax.set_xticks(range(len(methods)), [METHODS[m] for m in methods], fontsize=8)
        ax.set_xlim(-0.6, len(methods) - 0.4)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("SRKW test score")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c + "40", edgecolor=c, linewidth=1.5) for _, c in METRICS.values()]
    fig.legend(handles, [n for n, _ in METRICS.values()], loc="upper center", ncol=len(METRICS), frameon=False,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"SRKW vs rest on the comparison's test clips, 5 folds, {rule} decision (* uses target-site labels; "
                 "AdaBN / zero-shot = exp_2 effnet_b1)", x=0.01, ha="left", y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
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
    adabn = pd.read_csv(args.adabn_predictions).rename(columns={"variant": "method"})
    exp3 = exp3_predictions()
    exp3 = exp3[exp3.method == "finetune"].assign(method="exp3_effnet_b1_ft")
    preds = restrict_to_folds(pd.concat([adabn, exp3, comparison_predictions(missing)]))

    rows = []
    for (model, site, fold, method), pred in preds.groupby(["model", "site", "fold", "method"]):
        rows += [dict(model=model, site=site, fold=fold, method=method, **r) for r in scores(pred)]
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "metrics_per_fold.csv", index=False)

    # every method should be scored on the same number of test clips per site
    n_test = df.groupby(["site", "method", "model"]).n_test.agg(["min", "max"])
    expected = {s: int((pd.read_csv(str(SPLITS).format(dep=d)).fold_0 == "test").sum()) for s, (d, _) in SITES.items()}
    short = n_test[n_test["min"] < n_test.index.get_level_values("site").map(expected)]

    summary = (df.groupby(["site", "model", "method", "rule", "metric"]).value.agg(["mean", "std", "count"]).reset_index())
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    order = {m: i for i, m in enumerate(METHODS)}
    lines = ["# SRKW vs rest, comparison test clips, mean ± std over folds", ""]
    for site, (_, label) in SITES.items():
        s = summary[summary.site == site].copy()
        if s.empty:
            continue
        s["cell"] = s.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}" if r["count"] > 1 else f"{r['mean']:.3f}", axis=1)
        s["col"] = s.rule.where(s.rule != "any", "") + np.where(s.rule != "any", " ", "") + s.metric
        table = s.pivot_table(index=["method", "model"], columns="col", values="cell", aggfunc="first").reset_index()
        table = table.sort_values(["method", "model"], key=lambda c: c.map(order) if c.name == "method" else c)
        cols = ["method", "model", "pr_auc"] + [f"{r} {m}" for r in ("argmax", "val_tuned") for m in METRICS]
        table = table[[c for c in cols if c in table.columns]]
        lines += [f"## {label} ({expected[site]} test clips)", "", markdown(table), ""]
    if not short.empty:
        lines += ["## Methods scored on fewer test clips than expected", "", markdown(short.reset_index()), ""]
    if missing:
        lines += ["## Not included (missing / unfinished)", ""] + [f"- {m}" for m in missing] + [""]
    (args.out_dir / "summary.md").write_text("\n".join(lines))
    print(f"wrote {args.out_dir / 'summary.md'}")
    for rule in ("argmax", "val_tuned"):
        plot(df, rule, args.out_dir / f"srkw_{rule}.png")


if __name__ == "__main__":
    main()
