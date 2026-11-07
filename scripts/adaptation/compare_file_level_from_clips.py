"""File-level comparison built from the existing clip-level results: no new inference.

Every model's test (and val) clip predictions are grouped by audio file (LocalPath) and turned into one decision per
file, the way the detectors are used in deployment (a whole file gets one label). The comparison splits are grouped
by file (no file in two splits), and the test set is the same in all 5 folds, so each fold scores the same test files.

Methods
  site-trained, all three sites: Perch v2 probe (frozen), Perch v2 G (fine-tuned), EffNet-B1 ref (fine-tuned), from
      the newest FINISHED 100 % runs in MLflow; plus exp_2's zero-shot EffNet-B1.
  hallo-tuned-detectors, Strait of Georgia only (LimeKiln / Tekteksen files are in their training data): HALLO v1,
      BirdNET, sequence KW detector and the deployed cascade, from score_tuned_detectors.py's clip scores.

File truth: the set of its clips' labels. SRKW task: positive iff any clip is SRKW. KW task: iff any is SRKW or TKW.

File decisions
  pr_auc     threshold-free, on the file score = max over the file's clips (SRKW score, or KW score for the KW task).
  val_tuned  file score >= the threshold maximising file-level F1 on that fold's val files.
  any_clip   positive iff any clip's own decision is positive (argmax; BirdNET backend thresholds; sequence >= 0.75).
  vote       deployed BirdNET-style vote: file label = most common non-Background clip decision (ties -> higher mean
             score of that class); a file whose clips are all Background gets no label (negative).
  cascade    the resolver at file level: SRKW iff the sequence detector says KW for the file (any clip >= 0.75, its
             min_peaks=10 rule has no analogue on <= a few clips) and BirdNET's file decision (any / vote) is SRKW.

Caveat (printed in the summary): only annotated clips exist, not whole files. Strait of Georgia KW test files have one
clip each, so for them file level == clip level; its 18 negative files are dedicated background recordings
(118 clips each). LimeKiln has 14 test files and Tekteksen 15 (1 negative).

    python scripts/adaptation/compare_file_level_from_clips.py
"""
import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_label_budget import ANNOTATIONS
from compare_pretraining_results import COMPARISON_RUNS, SITES, TRACKING_URI, comparison_predictions, markdown, restrict_to_folds
from compare_tuned_detectors import RESULTS, SEQ_THRESH, best_threshold, detector_predictions, detector_scores

ARMS = ["perch_v2_probe", "perch_v2_supervised_source", "effnet_b1_ft"]
LABELS = {
    "perch_v2_probe": "Perch v2 frozen (probe)", "perch_v2_supervised_source": "Perch v2 fine-tuned (G)",
    "effnet_b1_ft": "EffNet-B1 ref (FT)", "zeroshot": "exp_2 EffNet-B1 zero-shot",
    "hallo_v1": "HALLO encoder v1", "birdnet": "BirdNET (Kait)", "sequence": "Sequence KW detector",
    "cascade": "Deployed cascade (seq ∧ BirdNET)",
}
DETECTOR_SITE = "strait_of_georgia"
TASKS = {"SRKW": {"SRKW"}, "KW": {"SRKW", "TKW"}}


def clip_kw_score(p: pd.DataFrame) -> pd.Series:
    m = p.method.iloc[0]
    if m == "sequence":
        return p.prob_KW
    if m == "birdnet":
        return p[["prob_SRKW", "prob_TKW"]].max(axis=1)
    return p.prob_SRKW + p.prob_TKW


def vote(group: pd.DataFrame) -> str:
    """most common non-Background clip decision; ties -> higher mean score of that class."""
    d = group[group.pred_label != "Background"]
    if d.empty:
        return "Background"
    counts = d.pred_label.value_counts()
    top = counts[counts == counts.max()].index
    if len(top) == 1:
        return top[0]
    return max(top, key=lambda c: d[f"prob_{c}"].mean() if f"prob_{c}" in d else 0.0)


def to_files(p: pd.DataFrame) -> pd.DataFrame:
    """one row per (fold, split, file) of a single method's clip rows."""
    p = p.assign(kw_score=clip_kw_score(p))
    g = p.groupby(["fold", "split", "LocalPath"])
    files = g.agg(labels=("true_label", lambda s: frozenset(s)), n_clips=("uid", "size"),
                  srkw_score=("prob_SRKW", "max"), kw_score=("kw_score", "max"))
    files["any_labels"] = g.pred_label.agg(lambda s: frozenset(s))
    files["vote_label"] = g.apply(vote, include_groups=False)
    return files.reset_index().assign(method=p.method.iloc[0])


def cascade_files(bn: pd.DataFrame, seq: pd.DataFrame) -> pd.DataFrame:
    keys = ["fold", "split", "LocalPath"]
    c = bn.merge(seq[keys + ["any_labels", "kw_score"]].rename(columns={"any_labels": "seq_labels", "kw_score": "seq_kw"}), on=keys)
    seq_kw = c.seq_labels.map(lambda s: "KW" in s)
    return c.assign(method="cascade",
                    srkw_score=c.srkw_score * c.seq_kw, kw_score=c.srkw_score * c.seq_kw,
                    any_labels=[frozenset({"SRKW"}) if k and "SRKW" in a else frozenset({"Background"})
                                for k, a in zip(seq_kw, c.any_labels)],
                    vote_label=np.where(seq_kw & (c.vote_label == "SRKW"), "SRKW", "Background")).drop(columns=["seq_labels", "seq_kw"])


def file_scores(f: pd.DataFrame) -> list[dict]:
    rows = []
    for task, positive in TASKS.items():
        y = f.labels.map(lambda s: bool(s & positive)).to_numpy()
        s = (f.srkw_score if task == "SRKW" else f.kw_score).to_numpy()
        val, test = (f.split == "val").to_numpy(), (f.split == "test").to_numpy()
        own = positive | {"KW"}  # the sequence detector's KW counts for both tasks, as in compare_tuned_detectors
        rules = {
            "val_tuned": s >= best_threshold(y[val], s[val]),
            "any_clip": f.any_labels.map(lambda a: bool(a & own)).to_numpy(),
            "vote": f.vote_label.isin(own).to_numpy(),
        }
        base = dict(task=task, n_files=int(test.sum()), n_pos=int(y[test].sum()))
        rows.append(base | dict(rule="any", metric="pr_auc", value=average_precision_score(y[test], s[test])))
        for rule, d in rules.items():
            rows += [base | dict(rule=rule, metric=n, value=fn(y[test], d[test], zero_division=0))
                     for n, fn in [("precision", precision_score), ("recall", recall_score), ("f1", f1_score)]]
            neg = test & ~y
            rows.append(base | dict(rule=rule, metric="fp_files", value=float(d[neg].sum())))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RESULTS / "file_level_from_clips")
    parser.add_argument("--detector-dir", type=Path, help="score_tuned_detectors.py csvs instead of MLflow")
    args = parser.parse_args()
    mlflow.set_tracking_uri(TRACKING_URI)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    arms = comparison_predictions(missing, {p: m for p, m in COMPARISON_RUNS.items() if m in ARMS})
    zs = pd.read_csv(RESULTS / "exp3_splits/adabn_exp3_predictions.csv")
    zs = zs[(zs.variant == "zeroshot") & (zs.model == "effnet_b1")].rename(columns={"variant": "method"})
    clips = restrict_to_folds(pd.concat([arms, zs], ignore_index=True))

    paths = pd.concat([pd.read_csv(str(ANNOTATIONS).format(dep=dep), usecols=["uid", "LocalPath"], low_memory=False).assign(site=site)
                       for site, (dep, _) in SITES.items()])
    folds = clips[(clips.method == "effnet_b1_ft") & (clips.site == DETECTOR_SITE)][["uid", "fold", "split", "true_label", "site"]]
    det = detector_predictions(detector_scores(DETECTOR_SITE, args.detector_dir)[0], folds.drop(columns="site")).assign(site=DETECTOR_SITE)
    clips = pd.concat([clips, det], ignore_index=True).merge(paths, on=["uid", "site"], how="left")
    assert clips.LocalPath.notna().all(), "clips without a file path"

    files = []
    for (site, method), p in clips[clips.method != "cascade"].groupby(["site", "method"]):
        files.append(to_files(p).assign(site=site))
    files = pd.concat(files, ignore_index=True)
    sog = files[files.site == DETECTOR_SITE]
    files = pd.concat([files, cascade_files(sog[sog.method == "birdnet"], sog[sog.method == "sequence"]).assign(site=DETECTOR_SITE)],
                      ignore_index=True)
    files.drop(columns=["labels", "any_labels"]).assign(labels=files.labels.map(lambda s: "+".join(sorted(s)))).to_csv(
        args.out_dir / "files.csv", index=False)

    df = pd.DataFrame([dict(site=s, method=m, fold=k, **r) for (s, m, k), f in files.groupby(["site", "method", "fold"])
                       for r in file_scores(f)])
    df.to_csv(args.out_dir / "metrics_per_fold.csv", index=False)

    lines = ["# File-level comparison from existing clip predictions", "",
             "Each file's score = max over its annotated clips; decisions: `val_tuned` (file threshold tuned on the fold's "
             "val files), `any_clip` (any clip's own decision positive), `vote` (majority of non-Background clip decisions, "
             "BirdNET's deployed file rule). Mean ± std over 5 folds; the test files are the same in every fold. "
             "`fp_files` = negative test files called positive.", "",
             "**Caveat:** only annotated clips exist, not whole files. Strait of Georgia KW files hold 1 clip each (file "
             "level == clip level for them); its negatives are 18 dedicated background files of 118 clips. LimeKiln has 14 "
             "test files, Tekteksen 15 (1 negative). Off-the-shelf detectors only at Strait of Georgia.", ""]
    for site, (_, site_label) in SITES.items():
        for task in TASKS:
            d = df[(df.site == site) & (df.task == task)]
            if d.empty:
                continue
            n = d.iloc[0]
            s = d.groupby(["method", "rule", "metric"]).value.agg(["mean", "std"]).reset_index()
            s["cell"] = np.where(s.metric == "fp_files", s["mean"].map("{:.1f}".format),
                                 s["mean"].map("{:.3f}".format) + " ± " + s["std"].fillna(0).map("{:.3f}".format))
            s["col"] = np.where(s.rule == "any", s.metric, s.rule + " " + s.metric)
            t = s.pivot(index="method", columns="col", values="cell")
            cols = ["pr_auc"] + [f"{r} {m}" for r in ("val_tuned", "any_clip", "vote") for m in ("f1", "precision", "recall", "fp_files")]
            t = t.reindex([m for m in LABELS if m in t.index])[cols]
            t.insert(0, "method", [LABELS[m] for m in t.index])
            lines += [f"## {site_label}: {task} vs rest ({n.n_files} test files, {n.n_pos} positive, "
                      f"{n.n_files - n.n_pos} negative)", "", markdown(t.reset_index(drop=True)), ""]
    if missing:
        lines += ["## Missing", ""] + [f"- {m}" for m in missing] + [""]
    (args.out_dir / "summary.md").write_text("\n".join(lines))
    print(f"wrote {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
