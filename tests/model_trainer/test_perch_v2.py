from pathlib import Path

import numpy as np
import pytest
import torch

from encoder_pipeline.model_trainer.perch_v2 import (
    EMBEDDING_DIM, WINDOW_SAMPLES, PerchV2Backbone, PerchV2Frontend, load_perch_v2_backbone, perch_v2_state_dict,
)

CHECKPOINT = Path(__file__).resolve().parents[2] / "data/model_trainer/pretrained/perch_v2_backbone.pt"


def _flax_variables(model: PerchV2Backbone) -> dict[str, np.ndarray]:
    """Invert perch_v2_state_dict: the SavedModel-style variable names / layouts for the model's current weights."""
    out, pre = {}, ".embedding_model.backbone."
    state = {k: v.numpy() for k, v in model.state_dict().items()}

    def conv(t, f):
        out[f"params{pre}{f}.kernel"] = state[f"{t}.weight"].transpose(2, 3, 1, 0)

    def bn(t, f):
        out[f"params{pre}{f}.scale"], out[f"params{pre}{f}.bias"] = state[f"{t}.weight"], state[f"{t}.bias"]
        out[f"batch_stats{pre}{f}.mean"], out[f"batch_stats{pre}{f}.var"] = state[f"{t}.running_mean"], state[f"{t}.running_var"]

    def dense(t, f):
        out[f"params{pre}{f}.kernel"], out[f"params{pre}{f}.bias"] = state[f"{t}.weight"].T, state[f"{t}.bias"]

    conv("stem.conv", "Stem_0.Conv_0"), bn("stem.bn", "Stem_0.BatchNorm_0")
    conv("head.conv", "Head_0.Conv_0"), bn("head.bn", "Head_0.BatchNorm_0")
    for i, block in enumerate(model.blocks):
        t, f = f"blocks.{i}", f"MBConv_{i}"
        if block.expand_conv is not None:
            conv(f"{t}.expand_conv", f"{f}.ExpandConv"), bn(f"{t}.expand_bn", f"{f}.ExpandBatchNorm")
        conv(f"{t}.depthwise_conv", f"{f}.DepthwiseConv"), bn(f"{t}.depthwise_bn", f"{f}.DepthwiseBatchNorm")
        dense(f"{t}.se.reduce", f"{f}.SqueezeAndExcitation_0.Reduce")
        dense(f"{t}.se.expand", f"{f}.SqueezeAndExcitation_0.Expand")
        conv(f"{t}.project_conv", f"{f}.ProjectConv"), bn(f"{t}.project_bn", f"{f}.ProjectBatchNorm")
    out["params.heads_protopnet_logits.bias"] = np.zeros(3, np.float32)  # non-backbone variables are ignored
    return out


def test_architecture_matches_perch_v2_b3():
    model = PerchV2Backbone()
    assert len(model.blocks) == 26
    assert sum(p.numel() for p in model.parameters()) == 10_695_512  # the SavedModel's backbone params
    assert [i for i, b in enumerate(model.blocks) if b.stride == 2] == [2, 5, 8, 18]


def test_frontend_and_backbone_shapes_and_layouts_agree():
    torch.manual_seed(0)
    audio = torch.randn(2, WINDOW_SAMPLES) * 0.1
    spec_tf = PerchV2Frontend(output_layout="time_freq")(audio)
    spec_ft = PerchV2Frontend(output_layout="freq_time")(audio)
    assert spec_tf.shape == (2, 1, 500, 128) and spec_ft.shape == (2, 1, 128, 500)
    assert torch.equal(spec_tf, spec_ft.transpose(-1, -2))

    a, b = PerchV2Backbone(input_layout="time_freq").eval(), PerchV2Backbone(input_layout="freq_time").eval()
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        assert a.spatial(spec_tf).shape == (2, EMBEDDING_DIM, 16, 4)
        assert torch.allclose(a(spec_tf), b(spec_ft), atol=1e-6)


def test_state_dict_mapping_round_trips_and_rejects_leftovers():
    torch.manual_seed(0)
    source = PerchV2Backbone()
    for p in source.state_dict().values():
        if p.is_floating_point():
            p.copy_(torch.randn_like(p))
    variables = _flax_variables(source)
    mapped = perch_v2_state_dict(variables)
    for k, v in source.state_dict().items():
        assert torch.equal(mapped[k], v), k

    variables["params.embedding_model.backbone.MBConv_26.ProjectConv.kernel"] = np.zeros((1, 1, 2, 2), np.float32)
    with pytest.raises(ValueError, match="not mapped"):
        perch_v2_state_dict(variables)


def test_gradients_reach_every_parameter():
    model = PerchV2Backbone(drop_path_rate=0.2).train()
    model(torch.randn(2, 1, 128, 500)).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="run scripts/perch/convert_perch_v2_to_torch.py first")
def test_converted_checkpoint_loads():
    model = load_perch_v2_backbone(str(CHECKPOINT)).eval()
    meta = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["meta"]
    assert meta["end-to-end embedding rel L2"] < 1e-4
    with torch.no_grad():
        emb = model(PerchV2Frontend()(torch.randn(1, WINDOW_SAMPLES) * 0.1))
    assert emb.shape == (1, EMBEDDING_DIM) and torch.isfinite(emb).all()


def test_freeze_batchnorm():
    frozen = PerchV2Backbone(freeze_batchnorm=True).train()
    bns = [m for m in frozen.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert frozen.training and all(not m.training for m in bns)
    before = bns[0].running_mean.clone()
    frozen(torch.randn(2, 1, 128, 120)).sum().backward()
    assert torch.equal(bns[0].running_mean, before) and bns[0].weight.grad is not None


def test_classifier_config_builds_perch_v2(tmp_path):
    from encoder_pipeline.model_trainer.config import ClassifierConfig
    from encoder_pipeline.model_trainer.models import ClassifierModel

    source = PerchV2Backbone()
    ckpt = tmp_path / "perch_v2_backbone.pt"
    torch.save({"state_dict": source.state_dict(), "meta": {}}, ckpt)
    config = ClassifierConfig.model_validate({"backbone_name": "perch_v2", "device": "cpu", "perch_v2": {
        "pretrained_path": str(ckpt), "freeze_batchnorm": True}})
    model = ClassifierModel(config, num_classes=4)
    assert isinstance(model.backbone, PerchV2Backbone)
    assert model.backbone.freeze_batchnorm
    assert torch.equal(model.backbone.stem.conv.weight, source.stem.conv.weight)
    assert model(torch.randn(2, 1, 128, 120)).shape == (2, 4)

    with pytest.raises(AssertionError, match="perch_v2 is required"):
        ClassifierModel(ClassifierConfig(backbone_name="perch_v2", device="cpu"), num_classes=4)
