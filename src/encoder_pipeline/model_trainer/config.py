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
    test_holdout_col: Optional[str] = None
    """Metadata column whose test_holdout_values pick the test split, e.g.
    'Dataset' to hold whole datasets out. Set together with
    test_holdout_values; test_size is then ignored and val_size is the
    fraction of the remaining (non-test) groups held out as val."""
    test_holdout_values: Optional[list[str]] = None
    """Values of test_holdout_col whose rows all go to test."""
    class_label_map: Optional[dict[str, str]] = None
    """Maps raw Labels values to collapsed labels before SpectrogramDataset
    builds its class list, e.g. {"SRKW": "KW", "TKW": "KW"}. Unmapped labels
    pass through unchanged."""
    classes: Optional[list[str]] = None
    """Fixed class list (post class_label_map), in index order, used instead of
    the classes present in the data -- e.g. to keep a fine-tune's head the same
    as its init_checkpoint's when some classes have no rows. Unset = sorted
    labels present in the HDF5."""
    splits_path: Optional[str] = None
    """Local path to a previously-logged splits.csv to reuse instead of
    computing a fresh split. Unset = compute_splits as usual."""
    val_from_train_size: float = 0.0
    """Fraction of train groups held out as val when a fold has none."""
    drop_unplaceable_windows: bool = False
    """Leave rows with a NaN FileBeginSec / Duration out of every split. HDF5s
    built before preprocessor/annotation.py raised on them hold these rows with
    their window at 0 s of the file instead of on the annotation."""
    train_overlap_tol_s: Optional[float] = None
    """Drop each train row whose window centre is within this many seconds of
    a train_overlap_splits row from another col_to_group_by group, e.g. the
    same call recorded on another channel of a hydrophone array. Window centre
    = the Soundfile name's start timestamp + FileBeginSec + Duration / 2.
    Unset = keep every train row."""
    train_overlap_splits: list[str] = ["test"]
    """Splits whose rows' simultaneous copies train_overlap_tol_s drops from train."""
    multi_label_window_s: Optional[float] = None
    """Set to the preprocessor's window length (2 * annotation.time_offset) to train multi-label: each row's target
    is every class annotated in the same file (LocalPath) within the window centred on the row (file-edge shifts
    ignored), with oversample_background_label set only when no other class is. Trains and probes with sigmoid +
    BCE; the oversampler still balances on each row's own label. Unset = one label per row (softmax)."""
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
SSAMBAVariant = Literal["ssamba_tiny", "ssamba_small", "ssamba_base"]


class SSAMBAEncoderConfig(StrictBaseModel):
    """Input + patch geometry of the SSAMBA (Vision Mamba) encoder"""

    input_fdim: int
    """Spectrogram frequency bins, e.g. the preprocessor's n_mels."""
    input_tdim: int
    """Spectrogram time frames."""
    fshape: int = 16
    """Patch height in frequency bins; patches don't overlap."""
    tshape: int = 16
    """Patch width in time frames; frames past the last whole patch are unused."""
    drop_path_rate: float = 0.1
    """Stochastic depth rate of the last block, ramped linearly from 0."""


class SSAMBAClassifierConfig(SSAMBAEncoderConfig):
    """SSAMBA encoder used as a classifier backbone"""

    pretrained_path: Optional[str] = None
    """Local path to a fold<n>_best.pt / fold<n>_last.pt written by a
    paradigm 'ssamba' run; its encoder weights are loaded before training.
    Needs the same variant and patch size; a different spectrogram size is
    fine (position embeddings are resized). Unset = train from scratch."""


AudioJEPAVariant = Literal["audio_jepa_vit_tiny", "audio_jepa_vit_small", "audio_jepa_vit_base"]
"""Audio-JEPA's VisionTransformer (model_trainer/audio_jepa, vendored from LudovicTuncay/Audio-JEPA) at ViT
tiny (192-d, 3 heads) / small (384-d, 6 heads) / base (768-d, 12 heads), 12 blocks each. The paper and the released
AudioSet checkpoint are base."""


class AudioJEPAEncoderConfig(StrictBaseModel):
    """Input + patch geometry of the Audio-JEPA ViT encoder. Spectrograms arrive (n_mels, frames) and are transposed to
    upstream's (time, freq) layout; frames / bins past the last whole patch are cropped."""

    input_fdim: int
    """Spectrogram frequency bins, e.g. the preprocessor's n_mels."""
    input_tdim: int
    """Spectrogram time frames."""
    patch_size: tuple[int, int] = (16, 16)
    """(time frames, freq bins) per patch, upstream's order; patches don't overlap."""
    drop_path_rate: float = 0.0
    """Stochastic depth of the last block, ramped linearly from 0 (upstream: 0)."""
    input_norm: Literal["per_clip", "none"] = "per_clip"
    """per_clip: standardise each spectrogram to zero mean / unit variance, as the SSAMBA arms do. none: feed the
    preprocessor's values as they are (upstream feeds raw log-fbank)."""


class AudioJEPAClassifierConfig(AudioJEPAEncoderConfig):
    """Audio-JEPA encoder used as a classifier backbone; the clip embedding is the mean patch token."""

    pretrained_path: Optional[str] = None
    """Local fold<n>_best.pt / fold<n>_last.pt of a paradigm 'audio_jepa' run; its downstream encoder (see
    AudioJEPAConfig.downstream_encoder) is loaded. Needs the same variant and patch size; another spectrogram size is
    fine (the position embeddings are fixed 2-D sin-cos and are rebuilt for this grid)."""
    audioset_checkpoint: Optional[str] = None
    """Local path to upstream's released AudioSet JEPA.ckpt (huggingface.co/ltuncay/Audio-JEPA, ViT-base); its
    audioset_checkpoint_module weights are loaded. Mutually exclusive with pretrained_path."""
    audioset_checkpoint_module: Literal["encoder", "target_encoder"] = "encoder"
    """Which of JEPA.ckpt's encoders to load; upstream's inference example loads 'encoder'."""


class AudioJEPAConfig(AudioJEPAEncoderConfig):
    """Audio-JEPA pretraining: context encoder, EMA target encoder, ViT predictor. Defaults are upstream's
    configs/model/jepa.yaml, configs/masks/random_block.yaml and MAWeightUpdate, except lr, which upstream sets for
    4 GPUs x batch 64 (ref_lr 1e-3 at batch 256)."""

    backbone_name: AudioJEPAVariant = "audio_jepa_vit_base"
    predictor_embed_dim: int = 384
    """VisionTransformerPredictor width (upstream default; its jepa.yaml doesn't override it)."""
    predictor_depth: int = 6
    predictor_num_heads: int = 12
    masks: Literal["random_block", "multiblock"] = "random_block"
    """random_block: upstream's default (masks/random_block.yaml), a random mask_ratio share of patches is
    predicted from the rest. multiblock: I-JEPA blocks (masks/multiblock.yaml)."""
    mask_ratio: tuple[float, float] = (0.4, 0.6)
    """random_block: share of patches to predict, drawn uniformly per batch."""
    enc_mask_scale: tuple[float, float] = (0.8, 1.0)
    """multiblock: context block size, as a share of the patches."""
    pred_mask_scale: tuple[float, float] = (0.15, 0.2)
    """multiblock: each target block's size, as a share of the patches."""
    aspect_ratio: tuple[float, float] = (0.5, 2.0)
    """multiblock: target block time / freq aspect ratio range."""
    npred: int = 3
    """multiblock: target blocks per clip."""
    loss_type: Literal["norm_mse", "mse", "smooth_l1"] = "norm_mse"
    norm_pix_loss: Union[bool, Literal["center"]] = True
    """Layer-normalise (True) or centre ('center') the target tokens before the loss."""
    ema_initial_tau: float = 0.996
    ema_final_tau: float = 1.0
    """Target-encoder EMA rate, cosine-ramped from initial to final over all training steps (MAWeightUpdate 'cos')."""
    downstream_encoder: Literal["target", "context"] = "target"
    """Encoder a fine-tune / embedding run loads from this run's checkpoint: the EMA target (I-JEPA's choice) or
    the gradient-trained context encoder."""
    audioset_checkpoint: Optional[str] = None
    """Start from upstream's released AudioSet JEPA.ckpt (ViT-base; encoder, target encoder and predictor are
    loaded) and continue pretraining. Unset = from scratch."""
    init_checkpoint: Optional[str] = None
    """Local fold<n>_best.pt / fold<n>_last.pt of a prior paradigm 'audio_jepa' run to continue from (whole model).
    Optimiser and schedules start fresh."""
    epochs: int = 10
    lr: float = 2.5e-4
    """Peak lr after warmup (upstream ref_lr 1e-3 at batch 256; 2.5e-4 is that linearly scaled to batch 64)."""
    start_lr: float = 1e-6
    final_lr: float = 0.0
    warmup_steps: int = Field(default=1000, ge=0)
    """WarmupCosineScheduler: linear start_lr -> lr over warmup_steps optimiser steps, then cosine to final_lr."""
    weight_decay: float = 1e-6
    final_weight_decay: float = 1e-6
    """CosineWDScheduler ref_wd -> final_wd (upstream: both 1e-6); biases and 1-d params get none."""
    betas: tuple[float, float] = (0.9, 0.95)
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    amp: bool = True
    """bf16 autocast (upstream trains bf16-mixed)."""
    max_grad_norm: Optional[float] = None
    """Global grad-norm clip; upstream doesn't clip."""
    early_stopping_patience: Optional[int] = Field(default=None, ge=1)
    log_train_loss_every: int = Field(default=0, ge=0)
    """When > 0, log mean train loss / lr / EMA tau every this-many optimiser steps (train_loss_step, lr_step,
    ema_tau_step)."""
    save_every_epoch: bool = False


PerchV2Variant = Literal["perch_v2"]


class PerchV2ClassifierConfig(StrictBaseModel):
    """Perch v2's EfficientNet-B3 (model_trainer/perch_v2.py) as a classifier backbone, initialised from Google's
    released weights. It sees the preprocessor's spectrograms exactly as they are, with no rescaling onto Perch's
    own log-mel units."""

    pretrained_path: str
    """Weights written by scripts/perch/convert_perch_v2_to_torch.py, e.g.
    data/model_trainer/pretrained/perch_v2_backbone.pt."""
    input_layout: Literal["freq_time", "time_freq"] = "freq_time"
    """Orientation of the spectrograms the loader yields: the preprocessor's are (n_mels, frames) = freq_time."""
    drop_path_rate: float = 0.0
    """Stochastic depth on the residual blocks while training; Perch itself was trained with 0.2."""
    freeze_batchnorm: bool = False
    """Keep every BatchNorm in eval mode (Perch's running statistics, no batch statistics) while training; the
    BatchNorm affine weights still train. Usual for small-data fine-tuning."""


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
    early_stopping_patience: Optional[int] = Field(default=None, ge=1)
    """Stop a fold once val loss has gone this many epochs without improving;
    None trains for all epochs. Requires a val loader."""


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
    early_stopping_patience: Optional[int] = Field(default=None, ge=1)
    """Stop a fold once val loss has gone this many epochs without improving;
    None trains for all epochs. Requires a val loader."""


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
    early_stopping_patience: Optional[int] = Field(default=None, ge=1)
    """Stop a fold once val loss has gone this many epochs without improving;
    None trains for all epochs. Requires a val loader."""


class SSAMBAConfig(SSAMBAEncoderConfig):
    """SSAMBA encoder + its masked spectrogram patch modelling heads"""

    backbone_name: SSAMBAVariant = "ssamba_tiny"
    mask_patch: int = 400
    """Patches masked per clip; must be below the clip's patch count."""
    cluster: bool = True
    """Mask square clusters of 3-5 patches a side instead of single patches."""
    mse_weight: float = 10.0
    """Loss = InfoNCE + mse_weight * reconstruction MSE."""
    init_checkpoint: Optional[str] = None
    """Local path to a fold<n>_best.pt / fold<n>_last.pt written by a prior
    paradigm 'ssamba' run; its weights (encoder + heads) are loaded before
    training, to continue pretraining. Optimizer and lr schedule start fresh.
    Needs the same variant, geometry and mask settings. Unset = train from
    scratch."""
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 5e-7
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    amp: bool = False
    """Mixed precision: bf16-autocast the forward / loss on self.device."""
    max_grad_norm: Optional[float] = None
    """Clip gradients to this global L2 norm before each optimizer step; None
    disables clipping. Independent of amp."""
    early_stopping_patience: Optional[int] = Field(default=None, ge=1)
    """Stop a fold once val loss has gone this many epochs without improving;
    None trains for all epochs. Requires a val loader."""
    lr_scheduler: bool = False
    """Cosine-anneal lr from config.lr to lr_scheduler_min_lr over config.epochs
    via torch.optim.lr_scheduler.CosineAnnealingLR."""
    lr_scheduler_min_lr: float = 0.0
    """Final lr at the end of cosine annealing; ignored when lr_scheduler is False."""
    lr_warmup_steps: int = Field(default=0, ge=0)
    """When > 0, replaces the per-epoch cosine (lr_scheduler is then ignored): lr rises linearly from 0 to config.lr
    over this many optimizer steps, then cosine-anneals to lr_scheduler_min_lr over the remaining
    epochs * steps_per_epoch steps, updated every optimizer step. 0 keeps the old behaviour."""
    log_train_loss_every: int = Field(default=0, ge=0)
    """When > 0, log the mean train loss of every this-many optimizer steps to MLflow as train_loss_step
    (step = global optimizer step), so within-epoch loss spikes are visible. 0 disables it."""
    save_every_epoch: bool = False
    """Also keep fold<n>_epoch<NN>.pt (same bundle as fold<n>_last.pt) after every epoch, in the run's
    data/model_trainer/<run_id>/ dir (not logged to MLflow), so intermediate checkpoints can be probed."""


class ClassifierConfig(StrictBaseModel):
    """Backbone + linear classification head"""

    backbone_name: Union[backbone, SSAMBAVariant, PerchV2Variant, AudioJEPAVariant] = "resnet18"
    ssamba: Optional[SSAMBAClassifierConfig] = None
    """Required when backbone_name is an ssamba_* variant."""
    audio_jepa: Optional[AudioJEPAClassifierConfig] = None
    """Required when backbone_name is an audio_jepa_* variant."""
    perch_v2: Optional[PerchV2ClassifierConfig] = None
    """Required when backbone_name is perch_v2."""
    init_checkpoint: Optional[str] = None
    """Local path to a fold<n>_best.pt / fold<n>_last.pt written by a prior
    classifier run; its weights (backbone + head) are loaded before training,
    for fine-tuning. Needs the same backbone_name and class list. Unset =
    train from scratch."""
    init_backbone_checkpoint: Optional[str] = None
    """Local path to any prior model_trainer checkpoint (classifier, SSL or ssamba run) whose backbone only is
    loaded before training; the head starts fresh, so the class list can differ. Needs the same backbone_name;
    an SSAMBA one may come from another spectrogram size (position embeddings are resized)."""
    augment: SpectrogramClassifierAugmentConfig = SpectrogramClassifierAugmentConfig()
    epochs: int = 10
    optimizer:Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
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
    early_stopping_patience: Optional[int] = Field(default=None, ge=1)
    """Stop a fold once val loss has gone this many epochs without improving;
    None trains for all epochs. Requires a val loader."""
    eval_every: Optional[int] = None
    """Every N epochs, log _evaluate's metrics (val / test) to mlflow Requires a val loader."""
    lr_scheduler: bool = False
    """Cosine-anneal lr from config.lr to lr_scheduler_min_lr over config.epochs
    via torch.optim.lr_scheduler.CosineAnnealingLR."""
    lr_scheduler_min_lr: float = 0.0
    """Final lr at the end of cosine annealing; ignored when lr_scheduler is False."""


class SearchParam(StrictBaseModel):
    """One hyperparameter's Ray Tune search domain: set exactly one field"""

    choice: Optional[list] = None
    grid_search: Optional[list] = None
    uniform: Optional[list] = None
    loguniform: Optional[list] = None
    randint: Optional[list] = None


class RayTuneConfig(StrictBaseModel):
    """Ray Tune search over the classifier / ssamba paradigm; consumed by
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
    max_epochs: Optional[int] = None
    """Epochs each trial trains for at most (ASHA's max_t); None = the
    paradigm's own epochs."""
    refit_best: bool = False
    """After the search, train the best trial's config for the paradigm's full
    epochs in the driver run, which then holds the final checkpoint."""


class ModelTrainerConfig(StrictBaseModel):
    run_name: Optional[str] = None
    """MLflow sub-run name. If unset, MLflow auto-generates one."""
    dataloader: DataLoaderConfig = DataLoaderConfig()
    paradigm: Literal["simclr", "moco", "moco_v3", "ssamba", "audio_jepa", "classifier"] = "simclr"
    """Which Trainer train_model runs."""
    simclr: SimCLRConfig = SimCLRConfig()
    moco: Optional[MoCoConfig] = None
    """Required when paradigm == 'moco'."""
    moco_v3: Optional[MoCoV3Config] = None
    """Required when paradigm == 'moco_v3'."""
    ssamba: Optional[SSAMBAConfig] = None
    """Required when paradigm == 'ssamba'."""
    audio_jepa: Optional[AudioJEPAConfig] = None
    """Required when paradigm == 'audio_jepa'."""
    classifier: Optional[ClassifierConfig] = None
    """Required when paradigm == 'classifier'."""
    tune: Optional[RayTuneConfig] = None
    """Ray Tune search config; when set with enabled=True the pipeline runs an
    HPO sweep"""
