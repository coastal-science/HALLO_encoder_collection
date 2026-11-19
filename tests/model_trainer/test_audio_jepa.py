import h5py
import mlflow
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from encoder_pipeline.model_trainer.audio_jepa_model import (
    AudioJEPABackbone, AudioJEPAModel, ema_tau, init_from_audioset,
)
from encoder_pipeline.model_trainer.config import (
    AudioJEPAClassifierConfig, AudioJEPAConfig, ClassifierConfig, ModelTrainerConfig,
)
from encoder_pipeline.model_trainer.data_loader import SpectrogramDataset
from encoder_pipeline.model_trainer.models import ClassifierModel, load_backbone
from encoder_pipeline.model_trainer.train import AudioJEPATrainer

# 8 time x 4 freq patches of 4 x 4 over a (n_mels=16, frames=34) spectrogram: 2 frames past the last patch are cropped
GEOMETRY = dict(input_fdim=16, input_tdim=34, patch_size=(4, 4))


def jepa_config(**overrides) -> AudioJEPAConfig:
    return AudioJEPAConfig(
        backbone_name="audio_jepa_vit_tiny", predictor_embed_dim=48, predictor_depth=1, predictor_num_heads=3,
        device="cpu", amp=False, **GEOMETRY, **overrides,
    )


def test_backbone_transposes_to_time_freq_crops_and_mean_pools():
    backbone = AudioJEPABackbone("audio_jepa_vit_tiny", AudioJEPAClassifierConfig(**GEOMETRY))
    specs = torch.randn(3, 1, 16, 34)
    assert backbone.prepare(specs).shape == (3, 1, 32, 16)
    assert (backbone.grid_t, backbone.grid_f, backbone.num_patches) == (8, 4, 32)
    assert backbone.tokens(specs).shape == (3, 32, 192)
    assert backbone(specs).shape == (3, 192)


def test_backbone_rejects_a_spectrogram_of_another_size():
    backbone = AudioJEPABackbone("audio_jepa_vit_tiny", AudioJEPAClassifierConfig(**GEOMETRY))
    with pytest.raises(AssertionError):
        backbone(torch.randn(2, 1, 16, 30))


def test_per_clip_norm_makes_the_backbone_invariant_to_gain_and_offset():
    backbone = AudioJEPABackbone("audio_jepa_vit_tiny", AudioJEPAClassifierConfig(**GEOMETRY)).eval()
    specs = torch.randn(2, 1, 16, 34)
    assert torch.allclose(backbone(specs), backbone(3 * specs + 7), atol=1e-4)


@pytest.mark.parametrize("masks", ["random_block", "multiblock"])
def test_jepa_step_trains_context_and_predictor_only(masks):
    # target blocks of 8-9 of the 32 patches: upstream's multiblock sampler retries forever below min_keep (4)
    trainer = AudioJEPATrainer(jepa_config(masks=masks, npred=2, pred_mask_scale=(0.25, 0.3)))
    specs = torch.randn(4, 1, 16, 34)
    collated = trainer.train_collator(specs)
    loss, z_pred = trainer.model(specs, collated.context_masks, collated.prediction_masks)
    assert torch.isfinite(loss)
    n_pred = sum(m.shape[1] for m in collated.prediction_masks)
    assert z_pred.shape[0] * z_pred.shape[1] == 4 * n_pred
    loss.backward()
    model = trainer.model
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert any(p.grad is not None for p in model.context_encoder.parameters())
    assert any(p.grad is not None for p in model.predictor.parameters())


def test_update_target_follows_the_ema_rule():
    model = AudioJEPAModel(jepa_config())
    with torch.no_grad():
        for p in model.context_encoder.parameters():
            p.add_(1.0)
    before = [p.clone() for p in model.target_encoder.parameters()]
    model.update_target(1.0)
    assert all(torch.equal(a, b) for a, b in zip(before, model.target_encoder.parameters()))
    model.update_target(0.0)
    assert all(torch.equal(a, b) for a, b in zip(model.context_encoder.parameters(), model.target_encoder.parameters()))


def test_ema_tau_ramps_cosine_from_initial_to_final():
    assert ema_tau(0, 100, 0.996, 1.0) == pytest.approx(0.996)
    assert ema_tau(50, 100, 0.996, 1.0) == pytest.approx(0.998)
    assert ema_tau(100, 100, 0.996, 1.0) == pytest.approx(1.0)


def test_backbone_property_picks_the_downstream_encoder():
    assert AudioJEPAModel(jepa_config()).backbone is not None
    model = AudioJEPAModel(jepa_config(downstream_encoder="context"))
    assert model.backbone is model.context_encoder
    model = AudioJEPAModel(jepa_config(downstream_encoder="target"))
    assert model.backbone is model.target_encoder


def test_schedulers_warm_up_then_decay_and_spare_biases_from_weight_decay():
    trainer = AudioJEPATrainer(jepa_config(lr=1e-3, start_lr=0.0, final_lr=1e-5, warmup_steps=10,
                                           weight_decay=0.04, final_weight_decay=0.4, epochs=2))
    trainer._build_schedulers(steps_per_epoch=50)
    lrs = []
    for _ in range(100):
        lrs.append(trainer.optimizer.param_groups[0]["lr"])
        trainer.lr_scheduler.step()
        trainer.wd_scheduler.step()
    assert lrs[0] == pytest.approx(0.0) and lrs[10] == pytest.approx(1e-3)
    assert lrs[5] == pytest.approx(5e-4)
    assert lrs[-1] == pytest.approx(1e-5, abs=1e-6)
    groups = trainer.optimizer.param_groups
    assert [g["weight_decay"] for g in groups[2:]] == [0.0, 0.0]
    assert groups[0]["weight_decay"] == groups[1]["weight_decay"] > 0.04


@pytest.fixture
def loaders(tmp_path):
    path = tmp_path / "dataset.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("spec", data=np.random.randn(12, 16, 34).astype(np.float32))
        h5.create_dataset("Labels", data=["a", "b"] * 6, dtype=h5py.string_dtype())
        h5.create_dataset("uid", data=np.arange(100, 112))
    dataset = SpectrogramDataset(str(path))
    return {
        "train": DataLoader(Subset(dataset, np.arange(8)), batch_size=4, shuffle=True),
        "val": DataLoader(Subset(dataset, np.arange(8, 12)), batch_size=4),
    }


def test_fit_logs_losses_and_its_checkpoint_starts_a_classifier_on_another_spectrogram_size(tmp_path, loaders):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    trainer = AudioJEPATrainer(jepa_config(epochs=2, warmup_steps=1, log_train_loss_every=1))
    with mlflow.start_run() as run:
        results = trainer.fit(loaders, fold=0, data_dir=str(tmp_path))
    assert np.isfinite(results["best_val_loss"])
    metrics = mlflow.MlflowClient().get_run(run.info.run_id).data.metrics
    assert {"fold0_train_loss", "fold0_val_loss", "val_pred_variance", "train_loss_step", "ema_tau_step"} <= set(metrics)
    assert trainer.global_step == 4 and trainer.tau > 0.996

    checkpoint = tmp_path / "model_trainer" / run.info.run_id / "fold0_best.pt"
    classifier = ClassifierModel(ClassifierConfig(
        backbone_name="audio_jepa_vit_tiny", device="cpu",
        audio_jepa=AudioJEPAClassifierConfig(input_fdim=16, input_tdim=50, patch_size=(4, 4),
                                             pretrained_path=str(checkpoint)),
    ), num_classes=3)
    source = torch.load(checkpoint, weights_only=False)["model"].target_encoder
    assert torch.equal(classifier.backbone.encoder.blocks[0].attn.qkv.weight, source.encoder.blocks[0].attn.qkv.weight)
    assert classifier.backbone.encoder.pos_embed.shape[1] == 12 * 4  # rebuilt for the longer clip's grid
    assert classifier(torch.randn(2, 1, 16, 50)).shape == (2, 3)


def test_val_scores_the_same_masks_every_epoch(loaders):
    trainer = AudioJEPATrainer(jepa_config())
    with mlflow.start_run():
        first = trainer._run_epoch(loaders["val"], train=False)
        second = trainer._run_epoch(loaders["val"], train=False)
    assert first == pytest.approx(second)


def test_init_from_audioset_loads_every_module_except_position_embeddings(tmp_path):
    source = AudioJEPAModel(jepa_config())
    state = {f"encoder.{k}": v for k, v in source.context_encoder.encoder.state_dict().items()}
    state |= {f"target_encoder.{k}": v + 1 for k, v in source.target_encoder.encoder.state_dict().items()}
    state |= {f"predictor.{k}": v for k, v in source.predictor.state_dict().items()}
    state["encoder.pos_embed"] = torch.zeros(1, 7, 192)  # another grid: ignored
    path = tmp_path / "JEPA.ckpt"
    torch.save({"state_dict": state}, path)

    model = AudioJEPAModel(jepa_config())
    init_from_audioset(model, str(path))
    qkv = "blocks.0.attn.qkv.weight"
    assert torch.equal(model.context_encoder.encoder.state_dict()[qkv], state[f"encoder.{qkv}"])
    assert torch.equal(model.target_encoder.encoder.state_dict()[qkv], state[f"target_encoder.{qkv}"])
    assert model.context_encoder.encoder.pos_embed.shape[1] == 32

    backbone = AudioJEPABackbone("audio_jepa_vit_tiny", AudioJEPAClassifierConfig(**GEOMETRY))
    init_from_audioset(backbone, str(path), "target_encoder")
    assert torch.equal(backbone.encoder.state_dict()[qkv], state[f"target_encoder.{qkv}"])


def test_paradigm_config_accepts_audio_jepa():
    config = ModelTrainerConfig(paradigm="audio_jepa", audio_jepa=jepa_config())
    assert config.audio_jepa.masks == "random_block" and config.audio_jepa.loss_type == "norm_mse"
