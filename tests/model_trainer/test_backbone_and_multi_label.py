"""init_backbone_checkpoint (backbone only, new head) and multi-label training targets / losses / metrics."""
import h5py
import numpy as np
import pytest
import torch

from encoder_pipeline.evaluation.linear_probe import LinearProbe
from encoder_pipeline.evaluation.metrics import classification_metrics, per_class_metrics
from encoder_pipeline.evaluation.predictions import predictions_frame
from encoder_pipeline.model_trainer.config import ClassifierConfig, SSAMBAClassifierConfig
from encoder_pipeline.model_trainer.data_loader import SpectrogramDataset
from encoder_pipeline.model_trainer.models import ClassifierModel, load_backbone
from encoder_pipeline.model_trainer.ssamba import SSAMBABackbone
from encoder_pipeline.model_trainer.train import ClassifierTrainer


def test_init_backbone_checkpoint_loads_backbone_with_a_new_head(tmp_path):
    source = ClassifierModel(ClassifierConfig(backbone_name="resnet18", device="cpu"), num_classes=4)
    path = tmp_path / "fold0_best.pt"
    torch.save({"model": source}, path)
    model = ClassifierModel(ClassifierConfig(backbone_name="resnet18", init_backbone_checkpoint=str(path)), num_classes=3)
    assert all(torch.equal(a, b) for a, b in zip(model.backbone.state_dict().values(), source.backbone.state_dict().values()))
    assert model.classifier.out_features == 3


def test_load_backbone_resizes_ssamba_position_embeddings(tmp_path):
    source = SSAMBABackbone("ssamba_tiny", SSAMBAClassifierConfig(input_fdim=64, input_tdim=128))  # 4 x 8 patches
    holder = torch.nn.Module()
    holder.backbone = source
    torch.save({"model": holder}, tmp_path / "ssl.pt")
    target = SSAMBABackbone("ssamba_tiny", SSAMBAClassifierConfig(input_fdim=65, input_tdim=235))  # 4 x 14 patches
    load_backbone(target, str(tmp_path / "ssl.pt"))
    assert target.v.pos_embed.shape == (1, 1 + 4 * 14, 192)
    assert torch.equal(target.v.pos_embed[:, 0], source.v.pos_embed[:, 0])  # cls position kept
    assert torch.equal(target.v.patch_embed.proj.weight, source.v.patch_embed.proj.weight)


def _windows_hdf5(tmp_path) -> str:
    # one file: a blue call at 10-20 s, a fin call at 22-23 s, background tiles at 60 s and 100 s; another file's fin
    path = tmp_path / "windows.h5"
    rows = [("a.wav", 10.0, 10.0, "BlueWhale"), ("a.wav", 22.0, 1.0, "FinWhale"), ("a.wav", 60.0, 5.0, "Background"),
            ("a.wav", 100.0, 5.0, "Background"), ("b.wav", 15.0, 1.0, "FinWhale")]
    with h5py.File(path, "w") as h5:
        h5.create_dataset("spec", data=np.zeros((len(rows), 2, 3)))
        for i, col in enumerate(["LocalPath", "FileBeginSec", "Duration", "Labels"]):
            h5.create_dataset(col, data=[str(r[i]) for r in rows], dtype=h5py.string_dtype())
    return str(path)


def test_multi_label_targets_include_every_class_in_the_window(tmp_path):
    path = _windows_hdf5(tmp_path)
    dataset = SpectrogramDataset(path, multi_label_window_s=30.0, background_label="Background")
    assert dataset.classes == ["Background", "BlueWhale", "FinWhale"]
    expected = [[0, 1, 1],  # blue window 0-30 s also holds the fin call
                [0, 1, 1],  # fin window 7.5-37.5 s also holds the blue call
                [1, 0, 0], [1, 0, 0],
                [0, 0, 1]]  # other file: no blue call leaks in
    np.testing.assert_array_equal(dataset.targets, expected)
    assert dataset.labels == [1, 2, 0, 0, 2]  # the oversampler still balances on each row's own label
    assert dataset[0][1].tolist() == [0, 1, 1]
    assert SpectrogramDataset(path).targets is None


def test_multi_label_trainer_uses_bce():
    trainer = ClassifierTrainer(ClassifierConfig(device="cpu"), num_classes=3, multi_label=True)
    assert isinstance(trainer.criterion, torch.nn.BCEWithLogitsLoss)


def test_multi_label_metrics_predictions_and_probe():
    y_true = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]])
    y_score = np.array([[0.1, 0.9, 0.8], [0.9, 0.2, 0.1], [0.2, 0.1, 0.4], [0.1, 0.7, 0.2]])
    y_pred = (y_score > 0.5).astype(int)
    metrics = classification_metrics(y_true, y_pred, y_score)
    assert metrics["pr_auc"] == pytest.approx(1.0) and metrics["recall"] == pytest.approx(5 / 6)
    assert per_class_metrics(y_true, y_pred, y_score, ["Background", "BlueWhale", "FinWhale"])["recall_FinWhale"] == 0.5
    frame = predictions_frame("test", None, y_true, y_score, ["Background", "BlueWhale", "FinWhale"])
    assert frame["true_label"].tolist()[0] == "BlueWhale;FinWhale" and frame["correct"].tolist() == [True, True, False, True]

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 8)).astype(np.float32)
    y = np.stack([x[:, 0] > 0, x[:, 1] > 0, x[:, 2] > 0], axis=1).astype(np.float32)
    probe_metrics, _, scores = LinearProbe(epochs=300, lr=0.05).evaluate({"train": (x, y), "test": (x, y)}, ["a", "b", "c"])
    assert scores["test"].shape == (200, 3) and probe_metrics["test_f1"] > 0.9
