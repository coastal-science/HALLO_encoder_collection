import pytest

from encoder_pipeline.model_trainer.config import ModelTrainerConfig, RayTuneConfig
from encoder_pipeline.model_trainer.hpo import apply_sample, build_search_space


@pytest.fixture
def base():
    return ModelTrainerConfig(paradigm="classifier", classifier={"device": "cpu"})


def test_apply_sample_merges_dotted_paths_and_revalidates(base):
    merged = apply_sample(base, {
        "classifier.optimizer": "sgd",
        "classifier.lr": 0.01,
        "classifier.augment.shift_frac": 0.25,
        "dataloader.batch_size": 64,
    })
    assert merged.classifier.optimizer == "sgd"
    assert merged.classifier.lr == 0.01
    assert merged.classifier.augment.shift_frac == 0.25
    assert merged.dataloader.batch_size == 64
    assert merged.classifier.weight_decay == base.classifier.weight_decay  # untouched fields preserved


def test_apply_sample_rejects_an_unknown_path(base):
    with pytest.raises(ValueError):
        apply_sample(base, {"classifier.lrr": 0.1})


def test_build_search_space_maps_each_domain_to_its_tune_sampler():
    tune = pytest.importorskip("ray.tune")
    space = build_search_space(RayTuneConfig(search_space={
        "classifier.optimizer": {"choice": ["adam", "sgd"]},
        "classifier.lr": {"loguniform": [1e-5, 0.01]},
        "classifier.epochs": {"randint": [10, 100]},
        "classifier.augment.shift_frac": {"uniform": [0.0, 0.3]},
        "dataloader.batch_size": {"grid_search": [32, 64]},
    }).search_space)

    assert set(space) == {
        "classifier.optimizer", "classifier.lr", "classifier.epochs",
        "classifier.augment.shift_frac", "dataloader.batch_size",
    }
    assert isinstance(space["classifier.lr"], tune.search.sample.Float)
    assert isinstance(space["classifier.epochs"], tune.search.sample.Integer)
    assert space["dataloader.batch_size"] == {"grid_search": [32, 64]}
