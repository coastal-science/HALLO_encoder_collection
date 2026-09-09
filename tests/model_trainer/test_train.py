import mlflow
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from encoder_pipeline.model_trainer.config import ClassifierConfig
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


@pytest.fixture
def loaders():
    specs, labels = torch.randn(8, 12, 20), torch.tensor([0, 1] * 4)
    return {
        "train": DataLoader(TensorDataset(specs, labels), batch_size=4),
        "val": DataLoader(TensorDataset(specs, labels), batch_size=4),
        "test": DataLoader(TensorDataset(specs, labels), batch_size=4),
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
