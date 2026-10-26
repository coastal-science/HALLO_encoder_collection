"""AdaBN (Li et al. 2016) on exp_2's held-out deployments, for exp_2's effnet_b1 and resnet18.

exp_2 trains on everything except four held-out Datasets, which form its single test split. Here
each held-out deployment is its own target domain: every BatchNorm layer's running mean / var is
re-estimated on that deployment's test clips (labels unused) and the adapted model is scored on
the same clips -- the standard transductive AdaBN setting. No weights are trained.

Variants, all scored on every test clip of the deployment:
  - zeroshot:        exp_2's checkpoint as-is (should reproduce exp_2's logged test predictions)
  - adabn_all:       BN stats from all of the deployment's test clips
  - adabn_background: BN stats from its Background clips only -- a proxy for adapting on the raw,
                     mostly-noise recording stream rather than on annotation-centred clips. Labels
                     pick the clips here, so it is not label-free on this data.

DORI VENUS (Saanich) is skipped: its test split is 4 TKW clips, too few for BN statistics.

Shares the GPU with other jobs: inference only, fp32, capped by --gpu-mem-gb.

    python scripts/adaptation/adabn.py [--out-dir scripts/adaptation/results]
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
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from encoder_pipeline.evaluation.predictions import predictions_frame
from encoder_pipeline.model_trainer.data_loader import SpectrogramDataset, load_saved_splits, load_uids

REPO = Path(__file__).resolve().parents[2]
TRACKING_URI = "http://localhost:5000"
EXPERIMENT = "dori_experiments"
RUNS = {"effnet_b1": "exp_2_effnet_b1", "resnet18": "exp_2_resnet18"}
CLASS_LABEL_MAP = {"AB": "Background", "UndBio": "Background"}  # exp_2's dataloader.class_label_map
DEPLOYMENTS = ["DORI VENUS (Strait of Georgia)", "LimeKiln", "Tekteksen"]
MIN_ADAPT_CLIPS = 100
SEED = 0


def exp2_run(run_name: str) -> tuple[str, str]:
    """(run_id, dataset_path) of the latest FINISHED exp_2 run named run_name."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT], order_by=["start_time DESC"],
        filter_string=f"tags.mlflow.runName = '{run_name}' and attributes.status = 'FINISHED'",
    )
    assert not runs.empty, f"no FINISHED run named {run_name}"
    return runs.iloc[0]["run_id"], runs.iloc[0]["params.dataset_path"]


def load_specs(hdf5_path: Path, rows: np.ndarray) -> torch.Tensor:
    """spec rows (sorted) as one CPU tensor, read in blocks."""
    with h5py.File(hdf5_path, "r") as h5:
        dset = h5["spec"]
        return torch.from_numpy(np.concatenate([dset[rows[i:i + 512]] for i in range(0, len(rows), 512)]))


@torch.no_grad()
def predict(model: nn.Module, specs: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    probs = [
        torch.softmax(model(specs[i:i + batch_size].to(device).unsqueeze(1)), dim=1).cpu()
        for i in range(0, len(specs), batch_size)
    ]
    return torch.cat(probs).numpy()


@torch.no_grad()
def adabn(model: nn.Module, specs: torch.Tensor, device: torch.device, batch_size: int) -> nn.Module:
    """Copy of model whose BatchNorm running stats are re-estimated on specs: a cumulative
    average over one shuffled pass (momentum=None), every other layer in eval mode so dropout /
    stochastic depth stay off."""
    adapted = copy.deepcopy(model).eval()
    bns = [m for m in adapted.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    assert bns, "model has no BatchNorm layers"
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    order = torch.randperm(len(specs), generator=torch.Generator().manual_seed(SEED))
    for i in range(0, len(order), batch_size):
        batch = order[i:i + batch_size]
        if len(batch) > 1:  # BN can't take batch stats of a single sample
            adapted(specs[batch].to(device).unsqueeze(1))
    return adapted.eval()


def scores(y_true: np.ndarray, probs: np.ndarray, classes: list[str]) -> list[dict]:
    """Per-class one-vs-rest precision / recall / F1 / PR-AUC for classes present in y_true, plus
    SRKW vs rest (SRKW iff the 4-class argmax is SRKW). Absent classes get no row: their metrics
    are undefined, but predictions into them still count against the present classes."""
    y_pred = probs.argmax(axis=1)
    targets = [(classes[c], y_true == c, y_pred == c, probs[:, c]) for c in range(len(classes))]
    srkw = classes.index("SRKW")
    targets.append(("SRKW_vs_rest", y_true == srkw, y_pred == srkw, probs[:, srkw]))
    rows = []
    for name, t, p, s in targets:
        if not t.any():
            continue
        rows.append(dict(
            target=name, n_pos=int(t.sum()), n=len(t),
            precision=precision_score(t, p, zero_division=0), recall=recall_score(t, p, zero_division=0),
            f1=f1_score(t, p, zero_division=0), pr_auc=average_precision_score(t, s),
        ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPO / "scripts/adaptation/results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu-mem-gb", type=float, default=4.0, help="cap on this process's GPU memory")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda", 0)
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_mem_gb / total_gb), device)

    mlflow.set_tracking_uri(TRACKING_URI)
    runs = {name: exp2_run(run_name) for name, run_name in RUNS.items()}
    dataset_paths = {dataset_path for _, dataset_path in runs.values()}
    assert len(dataset_paths) == 1, f"exp_2 runs use different datasets: {dataset_paths}"
    hdf5_path = REPO / dataset_paths.pop()

    dataset = SpectrogramDataset(str(hdf5_path), class_label_map=CLASS_LABEL_MAP)
    classes = dataset.classes
    labels = np.asarray(dataset.labels)
    uids = load_uids(str(hdf5_path))
    with h5py.File(hdf5_path, "r") as h5:
        deployment_of = h5["Dataset"].asstr()[:]

    models, test_rows = {}, {}
    for name, (run_id, _) in runs.items():
        run_dir = REPO / "data/model_trainer" / run_id
        model = torch.load(run_dir / "fold0_best.pt", map_location=device, weights_only=False)["model"]
        assert model.classifier.out_features == len(classes), f"{name}: head vs classes {classes}"
        models[name] = model.to(device).eval()
        (split,) = load_saved_splits(str(hdf5_path), str(run_dir / "splits.csv"))
        test_rows[name] = np.sort(split["test"])
    assert np.array_equal(*test_rows.values()), "exp_2 runs have different test splits"
    test_idx = next(iter(test_rows.values()))
    skipped = sorted(set(deployment_of[test_idx]) - set(DEPLOYMENTS))
    print(f"classes {classes}; skipping held-out {skipped} "
          f"({[(d, int((deployment_of[test_idx] == d).sum())) for d in skipped]} test clips)")

    pred_frames, metric_rows = [], []
    for deployment in DEPLOYMENTS:
        rows = test_idx[deployment_of[test_idx] == deployment]
        y_true = labels[rows]
        counts = {c: int((y_true == i).sum()) for i, c in enumerate(classes)}
        print(f"\n== {deployment}: {len(rows)} test clips {counts}")
        specs = load_specs(hdf5_path, rows)
        adapt_sets = {"adabn_all": np.ones(len(rows), dtype=bool), "adabn_background": y_true == classes.index("Background")}
        for name, model in models.items():
            variants = {"zeroshot": model}
            for variant, mask in adapt_sets.items():
                assert mask.sum() >= MIN_ADAPT_CLIPS, f"{deployment} {variant}: only {mask.sum()} clips"
                variants[variant] = adabn(model, specs[torch.from_numpy(mask)], device, args.batch_size)
            for variant, variant_model in variants.items():
                probs = predict(variant_model, specs, device, args.batch_size)
                frame = predictions_frame("test", uids[rows], y_true, probs, classes)
                frame.insert(0, "variant", variant)
                frame.insert(0, "deployment", deployment)
                frame.insert(0, "model", name)
                pred_frames.append(frame)
                for row in scores(y_true, probs, classes):
                    metric_rows.append(dict(model=name, deployment=deployment, variant=variant, **row))
                srkw = metric_rows[-1]
                print(f"  {name:9s} {variant:16s} SRKW-vs-rest P={srkw['precision']:.3f} R={srkw['recall']:.3f} "
                      f"F1={srkw['f1']:.3f} PR-AUC={srkw['pr_auc']:.3f}")
            del variants
        del specs
        torch.cuda.empty_cache()

    predictions = pd.concat(pred_frames)
    predictions.to_csv(args.out_dir / "adabn_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(args.out_dir / "adabn_metrics.csv", index=False)

    # zeroshot must reproduce exp_2's own logged test predictions
    for name, (run_id, _) in runs.items():
        logged = pd.read_csv(REPO / "data/model_trainer" / run_id / "fold0_predictions.csv")
        logged = logged[logged.split == "test"].set_index("uid")
        ours = predictions.query("model == @name and variant == 'zeroshot'").set_index("uid")
        prob_cols = [f"prob_{c}" for c in classes]
        diff = (ours[prob_cols] - logged.loc[ours.index, prob_cols]).abs().to_numpy().max()
        agree = (ours.pred_label == logged.loc[ours.index, "pred_label"]).mean()
        print(f"{name} zeroshot vs exp_2 logged test predictions: max |prob diff| {diff:.2e}, argmax agreement {agree:.4%}")
    print(f"\nwrote {args.out_dir}/adabn_predictions.csv and adabn_metrics.csv")


if __name__ == "__main__":
    main()
