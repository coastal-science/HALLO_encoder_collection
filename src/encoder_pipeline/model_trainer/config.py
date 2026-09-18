from typing import Literal, Optional, TypeAlias, Union

import torch
from pydantic import Field

from encoder_pipeline.common.base import StrictBaseModel


class DataLoaderConfig(StrictBaseModel):
    """torch DataLoader + split params for SpectrogramDataset."""

    batch_size: int = 32
    shuffle: bool = True
    n_folds: int = 1
    test_size: float = 0.1
    val_size: float = 0.1
    split_seed: int = 10
    num_workers: int = 0
    col_to_group_by: Optional[str] = None
    """Metadata column to split on, e.g. 'LocalPath' (file-level) or a
    deployment id column"""
    class_label_map: Optional[dict[str, str]] = None
    """Maps raw Labels values to collapsed labels before SpectrogramDataset
    builds its class list, e.g. {"SRKW": "KW", "TKW": "KW"}. Unmapped labels
    pass through unchanged."""
    splits_path: Optional[str] = None
    """Local path to a previously-logged splits.csv to reuse instead of
    computing a fresh split. Unset = compute_splits as usual."""
    val_from_train_size: float = 0.0
    """Fraction of train groups held out as val when a fold has none."""
    oversample: bool = True
    """Oversample the train split so every class is drawn to the largest
    class's size each epoch, via a WeightedRandomSampler. Applies to any
    paradigm."""
    oversample_background_label: Optional[str] = 'Background'
    """Label kept out of the oversampling target-size calc (e.g. 'Background').
    It's still drawn to the largest *other* class's size each epoch, so an
    outsized background pool is randomly downsampled rather than pulling every
    real class up to it."""


class SpectrogramSSLAugmentConfig(StrictBaseModel):
    time_mask_frac: float = 0.15
    freq_mask_frac: float = 0.15
    shift_frac: float = 0.1
    noise_std: float = 0.05


class SpectrogramClassifierAugmentConfig(StrictBaseModel):
    shift_frac: float = 0.1
    """Max circular time-window shift, as a fraction of the time axis length.
    Applied to train batches only."""


ResNetVariant = Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
EfficientNetVariant = Literal[
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
]
backbone: TypeAlias = Union[ResNetVariant, EfficientNetVariant]


class SimCLRConfig(StrictBaseModel):
    """Backbone + Lightly's SimCLR projection head"""

    backbone_name: backbone = "resnet18"
    augment: SpectrogramSSLAugmentConfig = SpectrogramSSLAugmentConfig()
    projection_hidden_dim: int = 512
    projection_out_dim: int = 128
    temperature: float = 0.5
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-6
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    amp: bool = False
    """Mixed precision: bf16-autocast the forward / loss on self.device."""
    max_grad_norm: Optional[float] = None
    """Clip gradients to this global L2 norm before each optimizer step; None
    disables clipping. Independent of amp."""


class MoCoConfig(StrictBaseModel):
    """Backbone + Lightly's MoCo projection head, momentum encoder, and
    memory-bank NTXent loss"""

    backbone_name: backbone = "resnet18"
    augment: SpectrogramSSLAugmentConfig = SpectrogramSSLAugmentConfig()
    projection_hidden_dim: int = 512
    projection_out_dim: int = 128
    temperature: float = 0.2
    memory_bank_size: int = 4096
    """Number of key embeddings kept as extra negatives."""
    momentum: float = 0.999
    """EMA rate for updating the momentum (key) encoder from the query encoder."""
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-6
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    amp: bool = False
    """Mixed precision: bf16-autocast the forward / loss on self.device."""
    max_grad_norm: Optional[float] = None
    """Clip gradients to this global L2 norm before each optimizer step; None
    disables clipping. Independent of amp."""


class MoCoV3Config(StrictBaseModel):
    """Backbone + Lightly's MoCo v3: 3-layer projection head, 2-layer
    prediction head, momentum encoder with cosine-annealed momentum, and a
    symmetric in-batch NTXent loss (no memory bank)."""

    backbone_name: backbone = "resnet18"
    augment: SpectrogramSSLAugmentConfig = SpectrogramSSLAugmentConfig()
    projection_hidden_dim: int = 4096
    projection_out_dim: int = 256
    prediction_hidden_dim: int = 4096
    temperature: float = 0.2
    momentum: float = 0.99
    """Base EMA rate for the key encoder; cosine-annealed to 1.0 over training."""
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-6
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    amp: bool = False
    """Mixed precision: bf16-autocast the forward / loss on self.device."""
    max_grad_norm: Optional[float] = None
    """Clip gradients to this global L2 norm before each optimizer step; None
    disables clipping. Independent of amp."""


class ClassifierConfig(StrictBaseModel):
    """Backbone + linear classification head"""

    backbone_name: backbone = "resnet18"
    augment: SpectrogramClassifierAugmentConfig = SpectrogramClassifierAugmentConfig()
    epochs: int = 10
    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    lr: float = 3e-4
    weight_decay: float = 1e-6
    momentum: float = 0.9
    """SGD / RMSprop momentum; ignored by adam / adamw."""
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    amp: bool = False
    """Mixed precision: bf16-autocast the forward / loss on self.device."""
    max_grad_norm: Optional[float] = None
    """Clip gradients to this global L2 norm before each optimizer step; None
    disables clipping. Independent of amp."""
    eval_every: Optional[int] = None
    """Every N epochs, log _evaluate's metrics (val / test) to mlflow Requires a val loader."""


class SearchParam(StrictBaseModel):
    """One hyperparameter's Ray Tune search domain: set exactly one field"""

    choice: Optional[list] = None
    grid_search: Optional[list] = None
    uniform: Optional[list] = None
    loguniform: Optional[list] = None
    randint: Optional[list] = None


class RayTuneConfig(StrictBaseModel):
    """Ray Tune search over the classifier paradigm; consumed by
    model_trainer.hpo.run_tuning when enabled."""

    enabled: bool = False
    search_space: dict[str, SearchParam] = {}
    """Dotted ModelTrainerConfig path -> SearchParam, e.g.
    {'classifier.lr': {loguniform: [1e-5, 1e-2]}}."""
    metric: str = "val_loss"
    """Reported metric ASHA and get_best_result rank on."""
    mode: Literal["min", "max"] = "min"
    num_samples: int = 16
    max_concurrent_trials: Optional[int] = None
    grace_period: int = 1
    """ASHA: min training_iteration before a trial can be stopped."""
    reduction_factor: int = 3
    """ASHA: keep 1/reduction_factor of trials at each rung."""
    resources_per_trial: dict[str, float] = {"cpu": 4, "gpu": 1}
    storage_path: Optional[str] = None
    """Ray Tune results dir; None -> <data_dir>/model_trainer/ray_tune."""


class ModelTrainerConfig(StrictBaseModel):
    run_name: Optional[str] = None
    """MLflow sub-run name. If unset, MLflow auto-generates one."""
    dataloader: DataLoaderConfig = DataLoaderConfig()
    paradigm: Literal["simclr", "moco", "moco_v3", "classifier"] = "simclr"
    """Which Trainer train_model runs."""
    simclr: SimCLRConfig = SimCLRConfig()
    moco: Optional[MoCoConfig] = None
    """Required when paradigm == 'moco'."""
    moco_v3: Optional[MoCoV3Config] = None
    """Required when paradigm == 'moco_v3'."""
    classifier: Optional[ClassifierConfig] = None
    """Required when paradigm == 'classifier'."""
    tune: Optional[RayTuneConfig] = None
    """Ray Tune search config; when set with enabled=True the pipeline runs an
    HPO sweep"""
