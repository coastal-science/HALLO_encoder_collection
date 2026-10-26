"""AdaBN on exp_3's 5-fold splits, so it is directly comparable with exp_3's fine-tune / linear probe.

For each exp_3 site (its own HDF5, uids and spectrograms identical to exp_2's dataset) and each
of its 5 folds, exp_2's effnet_b1 / resnet18 have their BatchNorm stats re-estimated on that fold's
TRAIN split and are scored on that fold's VAL split and exp_3's fixed TEST split -- the same clips
exp_3's fine-tune and linear probe are scored on. Val predictions are only for choosing a decision
threshold afterwards (plot_exp3_comparison.py); nothing is trained.

Variants (train split = the adaptation pool):
  - zeroshot:               exp_2's checkpoint as-is (the same for every fold; test is fixed)
  - adabn_train:            all train clips at their natural mix, labels unused
  - adabn_balanced:         train clips drawn by the pipeline's class_balanced_sampler -- the
                            sampler exp_2 trained with: every present class drawn equally, Background
                            downsampled to the largest other class, rare classes with replacement.
                            Uses train labels (oracle).
  - adabn_balanced_srkw_rest: 50/50 SRKW / non-SRKW train clips without replacement, the non-SRKW half
                            drawn from Background + HW + TKW at their natural mix -- the same SRKW vs
                            rest split the evaluation uses. Uses train labels (oracle).

    python scripts/adaptation/adabn_exp3.py [--out-dir ...] [--variants adabn_train ...] [--tag NAME]
"""

import argparse
import copy
from pathlib import Path

import h5py
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from adabn import CLASS_LABEL_MAP, EXPERIMENT, REPO, RUNS, TRACKING_URI, exp2_run, predict, scores
from encoder_pipeline.evaluation.predictions import predictions_frame
from encoder_pipeline.model_trainer.data_loader import (
    SpectrogramDataset, class_balanced_sampler, load_saved_splits, load_uids,
)

EXP3_PREFIX = "exp_3_effnet_b1_ft_"
CLASSES = ["Background", "HW", "SRKW", "TKW"]  # exp_3's dataloader.classes
SEED = 0


def exp3_runs() -> dict[str, tuple[str, str]]:
    """{site slug: (run_id, dataset_path)} of the FINISHED exp_3 runs."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string=f"tags.mlflow.runName LIKE '{EXP3_PREFIX}%' and attributes.status = 'FINISHED'",
    )
    return {
        row["tags.mlflow.runName"].removeprefix(EXP3_PREFIX).removesuffix("_5fold"): (row.run_id, row["params.dataset_path"])
        for _, row in runs.iterrows()
    }


@torch.no_grad()
def adabn(model: nn.Module, specs: torch.Tensor, rows: np.ndarray, device: torch.device, batch_size: int) -> nn.Module:
    """Copy of model whose BatchNorm running stats are a cumulative average (momentum=None) over
    one pass of specs[rows] in the given order; every other layer stays in eval mode."""
    adapted = copy.deepcopy(model).eval()
    for bn in (m for m in adapted.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)):
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        if len(batch) > 1:
            adapted(specs[torch.from_numpy(batch)].to(device).unsqueeze(1))
    return adapted.eval()


def adaptation_sets(train: np.ndarray, labels: np.ndarray, rng: np.random.Generator, seed: int) -> dict[str, np.ndarray]:
    """Row indices (already in pass order) of each variant's adaptation set."""
    srkw = CLASSES.index("SRKW")
    sampler = class_balanced_sampler(labels.tolist(), train, CLASSES, "Background")
    sampler.generator = torch.Generator().manual_seed(seed)
    balanced = train[np.fromiter(sampler, dtype=np.int64)]
    is_srkw = labels[train] == srkw
    n = min(is_srkw.sum(), (~is_srkw).sum())
    srkw_rest = np.concatenate([rng.choice(train[mask], n, replace=False) for mask in (is_srkw, ~is_srkw)])
    return {
        "adabn_train": rng.permutation(train),
        "adabn_balanced": balanced,  # WeightedRandomSampler draws are already in random order
        "adabn_balanced_srkw_rest": rng.permutation(srkw_rest),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPO / "scripts/adaptation/results/exp3_splits")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu-mem-gb", type=float, default=4.0, help="cap on this process's GPU memory")
    parser.add_argument("--variants", nargs="+", default=None, help="subset of variants to run (default: all)")
    parser.add_argument("--tag", default="adabn_exp3", help="output file prefix")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda", 0)
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_mem_gb / total_gb), device)

    mlflow.set_tracking_uri(TRACKING_URI)
    models = {}
    for name, run_name in RUNS.items():
        run_id, _ = exp2_run(run_name)
        bundle = torch.load(REPO / "data/model_trainer" / run_id / "fold0_best.pt", map_location=device, weights_only=False)
        models[name] = bundle["model"].to(device).eval()

    pred_frames, metric_rows = [], []
    for site, (run_id, dataset_path) in exp3_runs().items():
        hdf5_path = REPO / dataset_path
        dataset = SpectrogramDataset(str(hdf5_path), class_label_map=CLASS_LABEL_MAP, classes=CLASSES)
        labels, uids = np.asarray(dataset.labels), load_uids(str(hdf5_path))
        splits = load_saved_splits(
            str(hdf5_path), mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="splits.csv"),
        )
        test = np.sort(splits[0]["test"])
        assert all(np.array_equal(np.sort(s["test"]), test) for s in splits), f"{site}: test differs across folds"
        print(f"\n== {site}: {len(test)} test clips "
              f"{ {c: int((labels[test] == i).sum()) for i, c in enumerate(CLASSES)} }", flush=True)
        with h5py.File(hdf5_path, "r") as h5:
            specs = torch.from_numpy(h5["spec"][:])

        for name, model in models.items():
            zeroshot = predict(model, specs, device, args.batch_size)  # every site row, sliced per fold below
            for fold, split in enumerate(splits):
                eval_rows = {"val": np.sort(split["val"]), "test": test}
                rng = np.random.default_rng(SEED + fold)
                sets = adaptation_sets(np.sort(split["train"]), labels, rng, SEED + fold)
                if name == next(iter(models)):
                    print(f"  fold{fold} adaptation sets: " + ", ".join(
                        f"{v} n={len(r)} {np.bincount(labels[r], minlength=len(CLASSES)).tolist()}" for v, r in sets.items()
                    ), flush=True)
                variant_probs = {"zeroshot": {s: zeroshot[r] for s, r in eval_rows.items()}}
                if args.variants is not None:
                    variant_probs = {v: p for v, p in variant_probs.items() if v in args.variants}
                    sets = {v: r for v, r in sets.items() if v in args.variants}
                for variant, rows in sets.items():
                    adapted = adabn(model, specs, rows, device, args.batch_size)
                    variant_probs[variant] = {
                        s: predict(adapted, specs[torch.from_numpy(r)], device, args.batch_size) for s, r in eval_rows.items()
                    }
                for variant, split_probs in variant_probs.items():
                    for split_name, probs in split_probs.items():
                        r = eval_rows[split_name]
                        frame = predictions_frame(split_name, uids[r], labels[r], probs, CLASSES)
                        for col, value in [("variant", variant), ("fold", fold), ("site", site), ("model", name)]:
                            frame.insert(0, col, value)
                        pred_frames.append(frame)
                    test_scores = scores(labels[test], split_probs["test"], CLASSES)
                    metric_rows += [dict(model=name, site=site, fold=fold, variant=variant, **r) for r in test_scores]
                    srkw = test_scores[-1]
                    print(f"  {name:9s} fold{fold} {variant:22s} SRKW-vs-rest P={srkw['precision']:.3f} "
                          f"R={srkw['recall']:.3f} F1={srkw['f1']:.3f} PR-AUC={srkw['pr_auc']:.3f}", flush=True)
        del specs
        torch.cuda.empty_cache()

    pd.concat(pred_frames).to_csv(args.out_dir / f"{args.tag}_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(args.out_dir / f"{args.tag}_metrics.csv", index=False)
    print(f"\nwrote {args.out_dir}/{args.tag}_predictions.csv and {args.tag}_metrics.csv")


if __name__ == "__main__":
    main()
