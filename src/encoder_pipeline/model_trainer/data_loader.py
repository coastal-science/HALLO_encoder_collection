from pathlib import Path
from typing import Optional

import h5py
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from encoder_pipeline.common.file_utils import resolve_cpu_workers
from encoder_pipeline.model_trainer.config import DataLoaderConfig


class SpectrogramDataset(Dataset):
    def __init__(self, hdf5_path: str, label_col: str = "Labels", class_label_map: Optional[dict[str, str]] = None) -> None:
        self.hdf5_path = hdf5_path
        self._h5: Optional[h5py.File] = None

        with h5py.File(hdf5_path, "r") as h5:
            self.length = h5["spec"].shape[0]
            labels = h5[label_col].asstr()[:]
        if class_label_map is not None:
            labels = [class_label_map.get(label, label) for label in labels]
        self.classes = sorted(set(labels))
        self.label_to_idx = {label: i for i, label in enumerate(self.classes)}
        self.labels = [self.label_to_idx[label] for label in labels]

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        spec = self._file()["spec"][idx]
        return torch.from_numpy(spec).float(), self.labels[idx]


def compute_splits(hdf5_path: str, config: DataLoaderConfig) -> list[dict[str, np.ndarray]]:
    """
    Helper function to compute the train / test / val splits, and do k-fold splitting
    """
    with h5py.File(hdf5_path, "r") as h5:
        n = h5["spec"].shape[0]
        groups = h5[config.col_to_group_by].asstr()[:] if config.col_to_group_by else np.arange(n)
    unique_groups = np.unique(groups)

    def row_idx(group_subset: np.ndarray) -> np.ndarray:
        return np.where(np.isin(groups, group_subset))[0]

    if config.n_folds > 1:
        kfold = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.split_seed)
        return [
            {"train": row_idx(unique_groups[train_pos]), "val": row_idx(unique_groups[val_pos])}
            for train_pos, val_pos in kfold.split(unique_groups)
        ]

    remaining_groups, test_groups = unique_groups, np.array([])
    if config.test_size > 0:
        remaining_groups, test_groups = train_test_split(
            remaining_groups, test_size=config.test_size, random_state=config.split_seed,
        )

    train_groups, val_groups = remaining_groups, np.array([])
    if config.val_size > 0:
        val_frac = config.val_size / (1 - config.test_size)
        train_groups, val_groups = train_test_split(
            remaining_groups, test_size=val_frac, random_state=config.split_seed,
        )

    return [{"train": row_idx(train_groups), "val": row_idx(val_groups), "test": row_idx(test_groups)}]


def load_saved_splits(hdf5_path: str, splits_path: str, uid_col: str = "uid") -> list[dict[str, np.ndarray]]:
    """Loads a previously-saved splits.csv and maps its uid -> fold_N
    assignments onto this hdf5's row indices (by uid value, not row order,
    since a rebuilt hdf5 isn't guaranteed to keep the same row order). A saved
    uid the hdf5 no longer has -- e.g. dropped by dataset.classes_to_drop --
    is skipped."""
    with h5py.File(hdf5_path, "r") as h5:
        dset = h5[uid_col]
        uids = dset.asstr()[:] if h5py.check_string_dtype(dset.dtype) else dset[:]
    uid_to_row = {uid: i for i, uid in enumerate(uids)}

    saved = pd.read_csv(splits_path)
    fold_cols = sorted((c for c in saved.columns if c.startswith("fold_")), key=lambda c: int(c.split("_")[1]))

    splits = []
    for fold_col in fold_cols:
        split: dict[str, list[int]] = {}
        for uid, split_name in zip(saved["uid"], saved[fold_col]):
            if pd.isna(split_name) or uid not in uid_to_row:
                continue
            split.setdefault(split_name, []).append(uid_to_row[uid])
        splits.append({name: np.array(idx) for name, idx in split.items()})
    return splits


def holdout_val_from_train(
    hdf5_path: str, splits: list[dict[str, np.ndarray]], config: DataLoaderConfig,
) -> list[dict[str, np.ndarray]]:
    """Holds config.val_from_train_size of the train groups out as val for any
    fold lacking one, grouped by config.col_to_group_by."""
    if config.val_from_train_size <= 0 or all(len(split.get("val", ())) for split in splits):
        return splits

    with h5py.File(hdf5_path, "r") as h5:
        n = h5["spec"].shape[0]
        groups = h5[config.col_to_group_by].asstr()[:] if config.col_to_group_by else np.arange(n)

    out = []
    for split in splits:
        if len(split.get("val", ())):
            out.append(split)
            continue
        train_idx = split["train"]
        keep_groups, val_groups = train_test_split(
            np.unique(groups[train_idx]),
            test_size=config.val_from_train_size,
            random_state=config.split_seed,
        )
        out.append({
            **split,
            "train": train_idx[np.isin(groups[train_idx], keep_groups)],
            "val": train_idx[np.isin(groups[train_idx], val_groups)],
        })
    return out


def save_splits(hdf5_path: str, splits: list[dict[str, np.ndarray]], out_dir: str, uid_col: str = "uid") -> str:
    with h5py.File(hdf5_path, "r") as h5:
        dset = h5[uid_col]
        uids = dset.asstr()[:] if h5py.check_string_dtype(dset.dtype) else dset[:]

    df = pd.DataFrame({"uid": uids})
    for fold, split in enumerate(splits):
        col = np.empty(len(uids), dtype=object)
        for split_name, idx in split.items():
            col[idx] = split_name
        df[f"fold_{fold}"] = col

    out_path = Path(out_dir) / "splits.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return str(out_path)


def class_balanced_sampler(
    labels: list[int], subset_idx: np.ndarray, class_names: list[str], background_label: Optional[str] = None,
) -> WeightedRandomSampler:
    """WeightedRandomSampler over subset_idx positions that draws every present
    class to the same size each epoch: the largest class, or -- when
    background_label is given -- the largest class other than that one, so an
    outsized background pool is randomly downsampled to the real-class ceiling
    instead of raising it. Minority classes are oversampled with replacement."""
    subset_labels = np.asarray(labels)[subset_idx]
    counts = np.bincount(subset_labels, minlength=len(class_names))
    present = counts > 0
    ceiling = present.copy()
    if background_label in class_names:
        ceiling[class_names.index(background_label)] = False
    target = counts[ceiling if ceiling.any() else present].max()
    weights = 1.0 / counts[subset_labels]
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), int(target) * int(present.sum()), replacement=True,
    )


def build_dataloaders(hdf5_path: str, config: DataLoaderConfig, data_dir: str) -> list[dict[str, DataLoader]]:
    splits = load_saved_splits(hdf5_path, config.splits_path) if config.splits_path else compute_splits(hdf5_path, config)
    splits = holdout_val_from_train(hdf5_path, splits, config)
    splits_path = save_splits(hdf5_path, splits, out_dir=f"{data_dir}/model_trainer/{mlflow.active_run().info.run_id}")
    mlflow.log_artifact(splits_path)

    dataset = SpectrogramDataset(hdf5_path, class_label_map=config.class_label_map)

    def loader(name: str, idx: np.ndarray) -> DataLoader:
        # keep the GPU fed: parallel HDF5 reads, pinned host buffers, workers
        # kept alive across epochs (matters at 200 epochs).
        num_workers = resolve_cpu_workers(config.num_workers)
        mem_kwargs = dict(
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        if name == "train" and config.oversample:
            return DataLoader(
                Subset(dataset, idx), batch_size=config.batch_size,
                sampler=class_balanced_sampler(
                    dataset.labels, idx, dataset.classes, config.oversample_background_label,
                ),
                **mem_kwargs,
            )
        return DataLoader(
            Subset(dataset, idx), batch_size=config.batch_size, shuffle=config.shuffle, **mem_kwargs,
        )

    return [{name: loader(name, idx) for name, idx in split.items()} for split in splits]