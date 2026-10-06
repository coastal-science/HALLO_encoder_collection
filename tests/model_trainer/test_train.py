import h5py
import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from encoder_pipeline.model_trainer.config import ClassifierConfig
from encoder_pipeline.model_trainer.data_loader import SpectrogramDataset
from encoder_pipeline.model_trainer.train import ClassifierTrainer, build_optimizer


@pytest.mark.parametrize(
    ("name", "cls"),
    [("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW), ("sgd", torch.optim.SGD), ("rmsprop", torch.optim.RMSprop)],
)
def test_build_optimizer_returns_named_optimizer(name, cls):
    param = [torch.nn.Parameter(torch.zeros(2))]
    assert isinstance(build_optimizer(name, param, lr=1e-3, weight_decay=1e-6, momentum=0.9), cls)


def test_build_optimizer_passes_momentum_only_to_sgd_and_rmsprop():
    param = [torch.nn.Parameter(torch.zeros(2))]
    assert build_optimizer("sgd", param, 1e-3, 0.0, 0.8).param_groups[0]["momentum"] == 0.8
    assert "momentum" not in build_optimizer("adam", param, 1e-3, 0.0, 0.8).param_groups[0]


def test_build_optimizer_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_optimizer("nope", [torch.nn.Parameter(torch.zeros(2))], 1e-3, 0.0, 0.0)


def test_classifier_trainer_builds_the_configured_optimizer():
    trainer = ClassifierTrainer(ClassifierConfig(device="cpu", optimizer="sgd", momentum=0.95), num_classes=3)
    assert isinstance(trainer.optimizer, torch.optim.SGD)
    assert trainer.optimizer.param_groups[0]["momentum"] == 0.95


def test_classifier_trainer_scheduler_defaults_off():
    trainer = ClassifierTrainer(ClassifierConfig(device="cpu"), num_classes=3)
    assert trainer.scheduler is None


def test_classifier_trainer_scheduler_cosine_anneals_lr_to_min():
    config = ClassifierConfig(device="cpu", lr=1e-2, epochs=4, lr_scheduler=True, lr_scheduler_min_lr=1e-4)
    trainer = ClassifierTrainer(config, num_classes=3)
    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    for _ in range(config.epochs):
        trainer.scheduler.step()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(config.lr_scheduler_min_lr, abs=1e-6)


@pytest.fixture
def loaders(tmp_path):
    """8 train rows, then 4 val and 4 test rows, over one small hdf5."""
    path = tmp_path / "dataset.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("spec", data=np.random.randn(16, 12, 20).astype(np.float32))
        h5.create_dataset("Labels", data=["a", "b"] * 8, dtype=h5py.string_dtype())
        h5.create_dataset("uid", data=np.arange(100, 116))
    dataset = SpectrogramDataset(str(path))
    return {
        "train": DataLoader(Subset(dataset, np.arange(8)), batch_size=4),
        "val": DataLoader(Subset(dataset, np.arange(8, 12)), batch_size=4, shuffle=True),
        "test": DataLoader(Subset(dataset, np.arange(12, 16)), batch_size=4, shuffle=True),
    }


def test_fit_calls_on_epoch_end_once_per_epoch_and_returns_final_metrics(tmp_path, loaders):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    trainer = ClassifierTrainer(ClassifierConfig(device="cpu", epochs=3), num_classes=2)

    seen: list[tuple[int, dict]] = []
    with mlflow.start_run():
        results = trainer.fit(loaders, fold=0, data_dir=str(tmp_path), on_epoch_end=lambda e, m: seen.append((e, m)))

    assert [epoch for epoch, _ in seen] == [0, 1, 2]  # one call per epoch, last included
    assert "val_loss" in seen[0][1]
    assert {"best_val_loss", "val_f1", "test_f1"} <= results.keys()  # eval metrics merged into the return
    assert seen[-1][1] == results  # the final call carries the full metric dict


def test_fit_logs_the_best_epoch_and_a_per_sample_predictions_csv(tmp_path, loaders):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    trainer = ClassifierTrainer(ClassifierConfig(device="cpu", epochs=3), num_classes=2)

    with mlflow.start_run() as run:
        trainer.fit(loaders, fold=0, data_dir=str(tmp_path))

    client = mlflow.MlflowClient()
    val_losses = [m.value for m in client.get_metric_history(run.info.run_id, "fold0_val_loss")]
    assert client.get_run(run.info.run_id).data.metrics["fold0_best_epoch"] == int(np.argmin(val_losses))
    assert "fold0_predictions.csv" in [a.path for a in client.list_artifacts(run.info.run_id)]

    predictions = pd.read_csv(tmp_path / "model_trainer" / run.info.run_id / "fold0_predictions.csv")
    assert sorted(predictions.loc[predictions["split"] == "val", "uid"]) == [108, 109, 110, 111]
    assert sorted(predictions.loc[predictions["split"] == "test", "uid"]) == [112, 113, 114, 115]
    # uid % 2 picks the row's label, so a uid paired with another row's prediction would show up here
    assert list(predictions["true_label"]) == ["a" if uid % 2 == 0 else "b" for uid in predictions["uid"]]
    assert list(predictions["correct"]) == list(predictions["true_label"] == predictions["pred_label"])
    np.testing.assert_allclose(predictions[["prob_a", "prob_b"]].sum(axis=1), 1.0, rtol=1e-5)
