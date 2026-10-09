import h5py
import numpy as np
import pandas as pd

from encoder_pipeline.preprocessor.config import (
    AnnotationConfig, AudioFileConfig, DatasetConfig, PreprocessorConfig, SpectrogramConfig,
)
from encoder_pipeline.preprocessor.dataset import Dataset
from encoder_pipeline.preprocessor.samples import save_sample_images


def _write_csv(path, labels):
    n = len(labels)
    pd.DataFrame({
        "uid": range(n),
        "LocalPath": [f"/audio/f{i}.wav" for i in range(n)],
        "FileBeginSec": [0.0] * n,
        "FileEndSec": [1.0] * n,
        "Duration": [1.0] * n,
        "CenterTime": [0.5] * n,
        "Labels": labels,
    }).to_csv(path, index=False)
    return str(path)


def _dataset(csv, tmp_path, **ds_kwargs):
    return Dataset(
        SpectrogramConfig(), AudioFileConfig(), AnnotationConfig(),
        DatasetConfig(annotations_csv=csv, **ds_kwargs), data_dir=str(tmp_path),
    )


def test_load_annotations_drops_specified_classes(tmp_path):
    csv = _write_csv(tmp_path / "a.csv", ["HW", "SRKW", "Background", "HW", "SAR"])
    df = _dataset(csv, tmp_path, classes_to_drop=["Background", "SAR"])._load_annotations()

    assert sorted(df["Labels"]) == ["HW", "HW", "SRKW"]


def test_load_annotations_keeps_everything_when_classes_to_drop_unset(tmp_path):
    csv = _write_csv(tmp_path / "a.csv", ["HW", "Background"])
    assert len(_dataset(csv, tmp_path)._load_annotations()) == 2


def test_classes_to_drop_changes_the_output_hash(tmp_path):
    csv = _write_csv(tmp_path / "a.csv", ["HW", "Background"])
    plain = _dataset(csv, tmp_path)
    dropped = _dataset(csv, tmp_path, classes_to_drop=["Background"])

    assert plain.content_hash != dropped.content_hash


def test_save_sample_images_writes_one_grid_per_class(tmp_path):
    labels = ["HW"] * 5 + ["Background"] * 2
    with h5py.File(tmp_path / "d.h5", "w") as h5:
        h5.create_dataset("spec", data=np.random.default_rng(0).normal(size=(len(labels), 16, 20)).astype("float32"))
        h5.create_dataset("Labels", data=labels, dtype=h5py.string_dtype())
        h5.create_dataset("uid", data=np.arange(len(labels)))
    config = PreprocessorConfig(dataset=DatasetConfig(annotations_csv="unused.csv"))

    paths = save_sample_images(str(tmp_path / "d.h5"), config, tmp_path / "samples", n_per_class=3)

    assert sorted(p.name for p in paths) == ["Background.png", "HW.png"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_save_sample_images_skips_metadata_only_datasets(tmp_path):
    with h5py.File(tmp_path / "d.h5", "w") as h5:
        h5.create_dataset("Labels", data=["HW"], dtype=h5py.string_dtype())
    config = PreprocessorConfig(dataset=DatasetConfig(annotations_csv="unused.csv"))

    assert save_sample_images(str(tmp_path / "d.h5"), config, tmp_path / "samples", n_per_class=3) == []
