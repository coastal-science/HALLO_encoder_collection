"""Off-the-shelf hallo-tuned-detectors models vs the pretraining comparison's headline fine-tuned arms, on exactly the
same val / test clips of each fold (Strait of Georgia by default: LimeKiln / Tekteksen files are in the off-the-shelf
models' training data).

Inputs
  * arms: the newest FINISHED 100 % run of EffNet-B1 ref, SSAMBA A / B / C5 / D and the Perch v2 probe, from MLflow
    (compare_pretraining_results.comparison_predictions), plus exp_2's zero-shot EffNet-B1 (adabn_exp3_predictions.csv).
  * detectors: results/tuned_detectors/<detector>_<site>.csv from score_tuned_detectors.py (one score per clip; copied
    to every fold that holds the clip as val or test).

Detector scores and decisions
  hallo_v1   4-class softmax; argmax decision (same as the arms).
  birdnet    sigmoid SRKW / TKW / HW; decision = argmax class if its score >= the backend's threshold
             (configs/birdnet/base.json: SRKW 0.95, TKW 0.9, HW 0.9), else Background.
  sequence   KW score only (no ecotype): its SRKW-vs-rest score is its KW score; decision KW iff score >= 0.75
             (configs/hallo_seq_kw_det/base.json peak_height). Scored on KW-vs-rest too, where it belongs.
  cascade    the deployed resolver (src/resolve_killer_whale.py) at clip level: SRKW iff sequence says KW and BirdNET says
             SRKW; score = sequence KW score x BirdNET SRKW score. (The file-level min_peaks vote has no clip analogue.)

Tasks: SRKW vs rest (the headline) and KW vs rest (SRKW + TKW; arm / hallo_v1 score = prob_SRKW + prob_TKW, BirdNET =
max of its two KW sigmoids). Rules: argmax (the model's own decision) and val_tuned (threshold maximising F1 on the
fold's val clips). PR-AUC is threshold-free.

Detector scores come from the newest FINISHED, non-smoke "score_<detector>_<site>" run in MLflow experiment
hallo_tuned_detectors_comparison (score_tuned_detectors.py), or from --detector-dir csvs if given.
Writes <out-dir>/metrics_per_fold_<site>.csv, summary_<site>.md and the figures (dot plots, mean PR curves), and logs them
as run "compare_<site>" in the same experiment, with each method x task x rule x metric's fold mean / std as metrics
(e.g. birdnet.SRKW.val_tuned.f1_mean) and the MLflow run ids of every input as params.

    python scripts/adaptation/compare_tuned_detectors.py [--site strait_of_georgia]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score

from compare_pretraining_results import COMPARISON_RUNS, SITES, TRACKING_URI, comparison_predictions, markdown, restrict_to_folds
from plot_ssamba_vs_effnet import INK, MUTED, style

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "scripts/adaptation/results"
ARMS = ["effnet_b1_ft", "ssamba_A_supervised", "ssamba_B_ssl_source", "ssamba_C5_ssl_all_5ep", "ssamba_D_scratch", "perch_v2_probe"]
LABELS = {  # method -> label, in table / figure order: fine-tuned on the site first, then no site training
    "effnet_b1_ft": "EffNet-B1 ref (FT)", "ssamba_A_supervised": "SSAMBA A supervised (FT)",
    "ssamba_B_ssl_source": "SSAMBA B SSL source (FT)", "ssamba_C5_ssl_all_5ep": "SSAMBA C5 SSL all (FT)",
    "ssamba_D_scratch": "SSAMBA D scratch (FT)", "perch_v2_probe": "Perch v2 probe (site-trained)",
    "zeroshot": "exp_2 EffNet-B1 zero-shot", "hallo_v1": "HALLO encoder v1", "birdnet": "BirdNET (Kait)",
    "sequence": "Sequence KW detector", "cascade": "Deployed cascade (seq ∧ BirdNET)",
}
OFF_THE_SHELF = ["zeroshot", "hallo_v1", "birdnet", "sequence", "cascade"]
BIRDNET_THRESH = {"SRKW": 0.95, "TKW": 0.9, "HW": 0.9}
SEQ_THRESH = 0.75
GROUP_COLOUR = {"ft": "#2a78d6", "ots": "#eb6834"}  # dot plot: fine-tuned on site vs not trained on site
CURVES = {  # PR-curve figure: subset, colours from compare_label_budget's validated set; dashed = not site-trained
    "ssamba_A_supervised": ("#3f51b5", "-"), "perch_v2_probe": ("#b8336a", "-"),
    "hallo_v1": ("#00a3c4", (0, (5, 2))), "birdnet": ("#eda100", (0, (5, 2))), "sequence": ("#7a7a00", (0, (5, 2))),
}


EXPERIMENT = "hallo_tuned_detectors_comparison"
DETECTORS = ("hallo_v1", "birdnet", "sequence")


def detector_scores(site: str, detector_dir: Path | None) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """detector -> score_tuned_detectors.py csv, and detector -> the MLflow run id it came from."""
    if detector_dir is not None:
        return {n: pd.read_csv(detector_dir / f"{n}_{site}.csv") for n in DETECTORS}, {}
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    assert exp, f"no MLflow experiment {EXPERIMENT}: run score_tuned_detectors.py first"
    scores, ids = {}, {}
    for n in DETECTORS:
        runs = client.search_runs([exp.experiment_id], f"attributes.run_name = 'score_{n}_{site}' and attributes.status = 'FINISHED'",
                                  order_by=["attributes.start_time DESC"], max_results=1)
        assert runs, f"no FINISHED run score_{n}_{site} in {EXPERIMENT}"
        ids[n] = runs[0].info.run_id
        scores[n] = pd.read_csv(mlflow.artifacts.download_artifacts(run_id=ids[n], artifact_path=f"{n}_{site}.csv"))
    return scores, ids


def detector_predictions(d: dict[str, pd.DataFrame], folds: pd.DataFrame) -> pd.DataFrame:
    """The four detector 'methods' in the arms' prediction format, one row per (fold, uid) of `folds`."""
    d = {n: f.drop(columns=["LocalPath", "center_s"]) for n, f in d.items()}
    frames = []
    v1 = folds.merge(d["hallo_v1"], on="uid")
    probs = v1[[f"prob_{c}" for c in ("Background", "HW", "SRKW", "TKW")]].to_numpy()
    frames.append(v1.assign(method="hallo_v1", pred_label=np.array(["Background", "HW", "SRKW", "TKW"])[probs.argmax(1)]))

    bn = folds.merge(d["birdnet"], on="uid")
    top = bn[["prob_HW", "prob_SRKW", "prob_TKW"]].idxmax(axis=1).str.removeprefix("prob_")
    accepted = bn[["prob_HW", "prob_SRKW", "prob_TKW"]].max(axis=1) >= top.map(BIRDNET_THRESH)
    bn = bn.assign(method="birdnet", pred_label=top.where(accepted, "Background"))
    frames.append(bn)

    seq = folds.merge(d["sequence"], on="uid")
    frames.append(seq.assign(method="sequence", prob_SRKW=seq.prob_KW, pred_label=np.where(seq.prob_KW >= SEQ_THRESH, "KW", "Background")))

    cas = bn[["uid", "fold", "split", "true_label", "prob_SRKW", "pred_label"]].merge(
        seq[["uid", "fold", "split", "prob_KW"]], on=["uid", "fold", "split"])
    seq_kw = cas.prob_KW >= SEQ_THRESH
    frames.append(cas.assign(method="cascade", prob_SRKW=cas.prob_SRKW * cas.prob_KW,
                             pred_label=np.where(seq_kw & (cas.pred_label == "SRKW"), "SRKW", "Background"),
                             prob_KW=np.nan))
    return pd.concat(frames, ignore_index=True)


def kw_score(p: pd.DataFrame) -> pd.Series:
    if p.method.iloc[0] == "sequence":
        return p.prob_KW
    if p.method.iloc[0] == "birdnet":
        return p[["prob_SRKW", "prob_TKW"]].max(axis=1)
    if p.method.iloc[0] == "cascade":
        return p.prob_SRKW  # the cascade only ever emits SRKW
    return p.prob_SRKW + p.prob_TKW


def best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, s)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[np.argmax(f1)])


def scores(pred: pd.DataFrame) -> list[dict]:
    rows = []
    for task, positive, score in [("SRKW", {"SRKW"}, pred.prob_SRKW), ("KW", {"SRKW", "TKW"}, kw_score(pred))]:
        y = pred.true_label.isin(positive).to_numpy()
        val, test = (pred.split == "val").to_numpy(), (pred.split == "test").to_numpy()
        s = score.to_numpy()
        ok = ~np.isnan(s)
        decided = pred.pred_label.isin(positive | ({"KW"} if task == "KW" or pred.method.iloc[0] == "sequence" else set()))
        rules = {"argmax": decided.to_numpy(), "val_tuned": s >= best_threshold(y[val & ok], s[val & ok])}
        rows.append(dict(task=task, rule="any", metric="pr_auc", value=average_precision_score(y[test & ok], s[test & ok]),
                         n_test=int((test & ok).sum())))
        for rule, p in rules.items():
            rows += [dict(task=task, rule=rule, metric=name, value=fn(y[test], p[test], zero_division=0), n_test=int(test.sum()))
                     for name, fn in [("precision", precision_score), ("recall", recall_score), ("f1", f1_score)]]
    return rows


def dot_plot(df: pd.DataFrame, task: str, site_label: str, out: Path) -> None:
    panels = [("any", "pr_auc", "PR-AUC"), ("val_tuned", "f1", "F1, val-tuned threshold"), ("argmax", "f1", "F1, model's own decision"),
              ("argmax", "precision", "Precision, own decision"), ("argmax", "recall", "Recall, own decision")]
    methods = [m for m in LABELS if m in set(df.method)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels) + 2.6, 0.42 * len(methods) + 1.6), sharey=True)
    for ax, (rule, metric, title) in zip(axes, panels):
        for yi, m in enumerate(methods):
            v = df[(df.task == task) & (df.method == m) & (df.rule == rule) & (df.metric == metric)].value
            if v.empty:
                continue
            colour = GROUP_COLOUR["ots" if m in OFF_THE_SHELF else "ft"]
            ax.scatter(v, [yi] * len(v), s=16, color=colour, alpha=0.45, linewidth=0, zorder=2)
            ax.scatter([v.mean()], [yi], s=70, color=colour, edgecolor="white", linewidth=2, zorder=3,
                       marker="o" if m not in OFF_THE_SHELF else "D")
            ax.annotate(f"{v.mean():.2f}", (v.mean(), yi), xytext=(0, 7), textcoords="offset points", ha="center",
                        fontsize=7.5, color=INK)
        ax.axhline(len([m for m in methods if m not in OFF_THE_SHELF]) - 0.5, color=MUTED, linewidth=0.8)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlim(-0.02, 1.02)
        style(ax)
    axes[0].set_yticks(range(len(methods)), [LABELS[m] for m in methods], fontsize=9, color=INK)
    axes[0].invert_yaxis()
    fig.legend([Line2D([], [], color=GROUP_COLOUR["ft"], marker="o", linestyle="", markersize=7),
                Line2D([], [], color=GROUP_COLOUR["ots"], marker="D", linestyle="", markersize=7)],
               ["fine-tuned / trained on this site's train folds", "off the shelf (no site training)"],
               loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"{site_label}: {task} vs rest on the comparison's test clips — large mark = mean of 5 folds, small = folds",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (+ .pdf)")


def pr_plot(preds: pd.DataFrame, site_label: str, out: Path) -> None:
    grid = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, (task, positive) in zip(axes, [("SRKW", {"SRKW"}), ("KW", {"SRKW", "TKW"})]):
        handles = []
        for m, (colour, ls) in CURVES.items():
            curves, aps = [], []
            for _, p in preds[preds.method == m].groupby("fold"):
                t = p[p.split == "test"]
                s = (t.prob_SRKW if task == "SRKW" else kw_score(t)).to_numpy()
                y = t.true_label.isin(positive).to_numpy()
                ok = ~np.isnan(s)
                precision, recall, _ = precision_recall_curve(y[ok], s[ok])
                curves.append(np.interp(grid, recall[::-1], np.maximum.accumulate(precision)[::-1]))
                aps.append(average_precision_score(y[ok], s[ok]))
            if not curves:
                continue
            ax.plot(grid, np.mean(curves, axis=0), color=colour, linestyle=ls, linewidth=2)
            handles.append(Line2D([], [], color=colour, linestyle=ls, linewidth=2,
                                  label=f"{LABELS[m]}  {np.mean(aps):.2f} ± {np.std(aps):.2f}"))
        if task == "SRKW":  # deployed cascade: a binary decision -> one operating point per fold
            for _, p in preds[preds.method == "cascade"].groupby("fold"):
                t = p[p.split == "test"]
                y, d = (t.true_label == "SRKW").to_numpy(), (t.pred_label == "SRKW").to_numpy()
                ax.scatter(recall_score(y, d, zero_division=0), precision_score(y, d, zero_division=0), marker="X", s=60,
                           color=INK, edgecolor="white", linewidth=1, zorder=4)
            handles.append(Line2D([], [], color=INK, marker="X", linestyle="", markersize=8, label=LABELS["cascade"] + " (per fold)"))
        ax.legend(handles=handles, title="mean PR-AUC ± std over folds", title_fontsize=8, fontsize=8, loc="lower left",
                  frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, labelcolor=INK)
        ax.set_title(f"{task} vs rest", loc="left", fontsize=11, color=INK)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("recall", color=MUTED)
        style(ax)
    axes[0].set_ylabel("precision", color=MUTED)
    fig.suptitle(f"{site_label}: mean test PR curve over 5 folds (solid = trained on the site, dashed = off the shelf)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (+ .pdf)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=SITES, default="strait_of_georgia")
    parser.add_argument("--detector-dir", type=Path, help="read score_tuned_detectors.py csvs from here instead of MLflow")
    parser.add_argument("--no-mlflow", action="store_true", help="don't log the comparison run")
    parser.add_argument("--out-dir", type=Path, default=RESULTS / "tuned_detectors")
    args = parser.parse_args()
    mlflow.set_tracking_uri(TRACKING_URI)
    site_label = SITES[args.site][1]

    missing: list[str] = []
    arms = comparison_predictions(missing, {p: m for p, m in COMPARISON_RUNS.items() if m in ARMS})
    zeroshot = pd.read_csv(RESULTS / "exp3_splits/adabn_exp3_predictions.csv")
    zeroshot = zeroshot[(zeroshot.variant == "zeroshot") & (zeroshot.model == "effnet_b1")].rename(columns={"variant": "method"})
    arms = restrict_to_folds(pd.concat([arms, zeroshot]))
    arms = arms[arms.site == args.site]
    missing = [m for m in missing if args.site in m]

    # fold membership + labels from the arms' own rows, so detectors are scored on the identical clips
    folds = arms[arms.method == "effnet_b1_ft"][["uid", "fold", "split", "true_label"]]
    det, det_ids = detector_scores(args.site, args.detector_dir)
    preds = pd.concat([arms, detector_predictions(det, folds)], ignore_index=True)

    rows = [dict(method=m, fold=f, **r) for (m, f), p in preds.groupby(["method", "fold"]) for r in scores(p)]
    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / f"metrics_per_fold_{args.site}.csv", index=False)

    lines = [f"# {site_label}: off-the-shelf hallo-tuned-detectors vs fine-tuned arms", "",
             "Mean ± std over 5 folds, each fold's test clips (the arms' own). (FT) = fine-tuned on the site's train folds.",
             "The test clips are the same in every fold (fixed 10 % test, train / val rotate), so a model that is not trained "
             "on the site has std 0 except under val_tuned (its threshold comes from each fold's val clips).", ""]
    for task in ("SRKW", "KW"):
        s = df[df.task == task].groupby(["method", "rule", "metric"]).value.agg(["mean", "std"]).reset_index()
        s["cell"] = s["mean"].map("{:.3f}".format) + " ± " + s["std"].map("{:.3f}".format)
        s["col"] = np.where(s.rule == "any", s.metric, s.rule + " " + s.metric)
        t = s.pivot(index="method", columns="col", values="cell")
        cols = ["pr_auc"] + [f"{r} {m}" for r in ("val_tuned", "argmax") for m in ("f1", "precision", "recall")]
        t = t.reindex([m for m in LABELS if m in t.index])[cols]
        t.insert(0, "method", [LABELS[m] for m in t.index])
        n = df[(df.task == task) & (df.metric == "pr_auc")].groupby("method").n_test.min().reindex(t.index)
        t["min n_test"] = n.values
        lines += [f"## {task} vs rest", "", markdown(t.reset_index(drop=True)), ""]
    lines += ["`argmax` = the model's own decision (BirdNET: backend thresholds; sequence: KW score >= 0.75, counted as SRKW "
              "for the SRKW task; cascade: deployed resolver). `val_tuned` = F1-optimal threshold on that fold's val clips.", ""]
    if missing:
        lines += ["## Missing", ""] + [f"- {m}" for m in missing] + [""]
    (args.out_dir / f"summary_{args.site}.md").write_text("\n".join(lines))
    print(f"wrote {args.out_dir / f'summary_{args.site}.md'}")
    for task in ("SRKW", "KW"):
        dot_plot(df, task, site_label, args.out_dir / f"dots_{task.lower()}_{args.site}.png")
    pr_plot(preds, site_label, args.out_dir / f"pr_curves_{args.site}.png")
    if not args.no_mlflow:
        log_comparison(args, df, det_ids, arms, missing)


def log_comparison(args, df: pd.DataFrame, det_ids: dict[str, str], arms: pd.DataFrame, missing: list[str]) -> None:
    mlflow.set_experiment(EXPERIMENT)
    agg = df.groupby(["method", "task", "rule", "metric"]).value.agg(["mean", "std"]).reset_index()
    metrics = {}
    for r in agg.itertuples():
        key = f"{r.method}.{r.task}.{r.metric}" if r.rule == "any" else f"{r.method}.{r.task}.{r.rule}.{r.metric}"
        metrics[f"{key}_mean"], metrics[f"{key}_std"] = float(r.mean), float(0 if np.isnan(r.std) else r.std)
    with mlflow.start_run(run_name=f"compare_{args.site}"):
        mlflow.set_tags(dict(stage="compare", site=args.site))
        mlflow.log_params({f"input.{n}": i for n, i in det_ids.items()} | dict(
            site=args.site, arms=",".join(sorted(set(arms.method))), detector_source=str(args.detector_dir or "mlflow"),
            n_missing=len(missing)))
        mlflow.log_metrics(metrics)
        for f in sorted(args.out_dir.glob(f"*_{args.site}.*")):
            if f.suffix in {".md", ".csv", ".png", ".pdf"} and not f.name.startswith(DETECTORS):
                mlflow.log_artifact(str(f))
    print(f"logged MLflow run compare_{args.site} to experiment {EXPERIMENT} ({len(metrics)} metrics)")


if __name__ == "__main__":
    main()
