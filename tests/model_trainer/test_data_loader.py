import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from encoder_pipeline.model_trainer.config import DataLoaderConfig
from encoder_pipeline.model_trainer.data_loader import (
    class_balanced_sampler, compute_splits, holdout_val_from_train, load_saved_splits,
)


@pytest.fixture
def hdf5_path(tmp_path):
    """8 rows across 4 groups (2 rows each), so group-level splits are
    exercisable at a small scale."""
    path = tmp_path / "dataset.h5"
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    with h5py.File(path, "w") as h5:
        h5.create_dataset("spec", data=np.zeros((len(groups), 4)))
        h5.create_dataset("file_id", data=groups, dtype=h5py.string_dtype())
    return str(path)


def _groups(hdf5_path: str, col: str) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as h5:
        return h5[col].asstr()[:]


def test_train_test_split_covers_every_row_exactly_once(hdf5_path):
    config = DataLoaderConfig(n_folds=1, test_size=0.25, val_size=0.25, split_seed=0, col_to_group_by="file_id")
    [split] = compute_splits(hdf5_path, config)

    all_idx = sorted(np.concatenate([split["train"], split["val"], split["test"]]))
    assert all_idx == list(range(8))


def test_train_test_split_keeps_groups_together(hdf5_path):
    config = DataLoaderConfig(n_folds=1, test_size=0.25, val_size=0.25, split_seed=0, col_to_group_by="file_id")
    [split] = compute_splits(hdf5_path, config)
    groups = _groups(hdf5_path, "file_id")

    train_groups = set(groups[split["train"]])
    val_groups = set(groups[split["val"]])
    test_groups = set(groups[split["test"]])
    assert not train_groups & val_groups
    assert not train_groups & test_groups
    assert not val_groups & test_groups


def test_train_test_split_is_reproducible_with_same_seed(hdf5_path):
    config = DataLoaderConfig(n_folds=1, test_size=0.25, val_size=0.25, split_seed=42, col_to_group_by="file_id")
    [split_a] = compute_splits(hdf5_path, config)
    [split_b] = compute_splits(hdf5_path, config)

    for key in ("train", "val", "test"):
        assert list(split_a[key]) == list(split_b[key])


def test_kfold_puts_every_group_in_val_exactly_once(hdf5_path):
    config = DataLoaderConfig(n_folds=4, split_seed=0, col_to_group_by="file_id")
    folds = compute_splits(hdf5_path, config)
    groups = _groups(hdf5_path, "file_id")

    assert len(folds) == 4
    val_groups_per_fold = [set(groups[fold["val"]].tolist()) for fold in folds]
    total = sum(len(vg) for vg in val_groups_per_fold)
    union = set().union(*val_groups_per_fold)
    assert total == len(union) == len(set(groups))  # each group in exactly one fold's val set

    for fold in folds:
        assert not set(groups[fold["train"]]) & set(groups[fold["val"]])


def test_kfold_no_leakage_between_train_and_val(hdf5_path):
    config = DataLoaderConfig(n_folds=4, split_seed=0, col_to_group_by="file_id")
    folds = compute_splits(hdf5_path, config)

    for i, fold in enumerate(folds):
        assert not set(fold["train"]) & set(fold["val"]), f"fold {i} leaks rows between train and val"


def test_kfold_all_rows_accounted_for_in_every_fold(hdf5_path):
    config = DataLoaderConfig(n_folds=4, split_seed=0, col_to_group_by="file_id")
    folds = compute_splits(hdf5_path, config)

    for i, fold in enumerate(folds):
        all_idx = sorted(np.concatenate([fold["train"], fold["val"]]))
        assert all_idx == list(range(8)), f"fold {i} doesn't cover every row"


def test_no_col_to_group_by_splits_row_by_row(hdf5_path):
    config = DataLoaderConfig(n_folds=1, test_size=0.25, val_size=0.0, split_seed=0, col_to_group_by=None)
    [split] = compute_splits(hdf5_path, config)

    assert len(split["test"]) == 2
    assert len(split["train"]) + len(split["test"]) == 8


def test_holdout_val_from_train_carves_a_group_disjoint_val_and_leaves_test(hdf5_path):
    config = DataLoaderConfig(val_from_train_size=0.25, split_seed=0, col_to_group_by="file_id")
    test_idx = np.array([6, 7])
    [split] = holdout_val_from_train(hdf5_path, [{"train": np.arange(6), "test": test_idx}], config)
    groups = _groups(hdf5_path, "file_id")

    assert list(split["test"]) == list(test_idx)  # untouched
    assert sorted(np.concatenate([split["train"], split["val"]])) == list(range(6))
    assert not set(groups[split["train"]]) & set(groups[split["val"]])
    assert len(split["val"]) > 0


def test_holdout_val_from_train_is_a_noop_when_the_fold_already_has_val(hdf5_path):
    config = DataLoaderConfig(val_from_train_size=0.25, split_seed=0, col_to_group_by="file_id")
    folds = [{"train": np.arange(4), "val": np.array([4, 5]), "test": np.array([6, 7])}]
    assert holdout_val_from_train(hdf5_path, folds, config) is folds


def test_holdout_val_from_train_is_a_noop_when_disabled(hdf5_path):
    config = DataLoaderConfig(val_from_train_size=0.0, col_to_group_by="file_id")
    folds = [{"train": np.arange(6), "test": np.array([6, 7])}]
    assert holdout_val_from_train(hdf5_path, folds, config) is folds


def test_class_balanced_sampler_evens_out_a_skewed_train_split():
    labels = [0] * 90 + [1] * 10
    subset_idx = np.arange(len(labels))
    sampler = class_balanced_sampler(labels, subset_idx, class_names=["a", "b"])

    torch.manual_seed(0)
    drawn = np.asarray(labels)[list(sampler)]
    assert len(drawn) == 90 * 2  # every class drawn to the largest class's size
    assert 0.4 < drawn.mean() < 0.6  # both classes now roughly equally represented


def test_class_balanced_sampler_indexes_into_subset_positions_only():
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    subset_idx = np.array([1, 3, 5])  # all class 1
    drawn = list(class_balanced_sampler(labels, subset_idx, class_names=["a", "b"]))

    assert all(0 <= i < len(subset_idx) for i in drawn)


def test_load_saved_splits_skips_uids_the_hdf5_no_longer_has(tmp_path):
    """A dropped class (dataset.classes_to_drop) removes its uids from the
    hdf5, but the saved splits.csv still lists them."""
    path = tmp_path / "d.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("spec", data=np.zeros((3, 4)))
        h5.create_dataset("uid", data=np.array([10, 11, 12]))  # uid 13, 14 were dropped

    splits_csv = tmp_path / "splits.csv"
    pd.DataFrame({"uid": [10, 11, 12, 13, 14], "fold_0": ["train", "train", "test", "train", "test"]}).to_csv(
        splits_csv, index=False,
    )

    [split] = load_saved_splits(str(path), str(splits_csv))
    assert sorted(split["train"]) == [0, 1]  # rows for uid 10, 11; uid 13 skipped
    assert sorted(split["test"]) == [2]      # row for uid 12; uid 14 skipped


def test_class_balanced_sampler_downsamples_background_to_the_largest_real_class():
    labels = [0] * 1000 + [1] * 100 + [2] * 20  # 0 = background, huge
    class_names = ["background", "hw", "kw"]
    subset_idx = np.arange(len(labels))

    torch.manual_seed(0)
    drawn = np.asarray(labels)[list(class_balanced_sampler(labels, subset_idx, class_names, "background"))]
    counts = np.bincount(drawn, minlength=3)

    assert len(drawn) == 100 * 3  # target is hw's 100, not background's 1000
    assert all(abs(c - 100) < 40 for c in counts)  # background pulled down, kw pulled up


def test_class_balanced_sampler_draws_a_fresh_background_subset_each_epoch():
    labels = [0] * 1000 + [1] * 100
    sampler = class_balanced_sampler(labels, np.arange(1100), ["background", "hw"], "background")

    torch.manual_seed(1)
    epoch1 = set(sampler)
    torch.manual_seed(2)
    epoch2 = set(sampler)
    assert epoch1 != epoch2  # different random background windows drawn
