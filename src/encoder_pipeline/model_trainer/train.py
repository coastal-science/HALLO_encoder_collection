import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from lightly.loss import NTXentLoss
from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule
from torch.utils.data import DataLoader
from tqdm import tqdm
from encoder_pipeline.evaluation.metrics import classification_metrics
from encoder_pipeline.model_trainer.config import (
    ClassifierConfig, MoCoConfig, MoCoV3Config, ModelTrainerConfig, SimCLRConfig,
)
from encoder_pipeline.model_trainer.models import ClassifierModel, MoCoModel, MoCoV3Model, SimCLRModel
from encoder_pipeline.model_trainer.augment import SpectrogramClassifierAugment, SpectrogramSSLAugment
from encoder_pipeline.preprocessor.config import SpectrogramConfig


def build_optimizer(
    name: str, params, lr: float, weight_decay: float, momentum: float = 0.0,
) -> torch.optim.Optimizer:
    """torch optimizer by name; momentum applies to sgd / rmsprop only."""
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"unknown optimizer {name!r}")


class Trainer(ABC):
    model: nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device
    epochs: int
    amp: bool = False
    max_grad_norm: Optional[float] = None
    eval_every: Optional[int] = None
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None

    def _autocast(self) -> torch.autocast:
        """bf16 autocast on self.device when self.amp is set, else a no-op."""
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp)

    def _optimizer_step(self, loss: torch.Tensor) -> None:
        """Backward, optional global grad-norm clip, then optimizer step."""
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def fit(
        self, loaders: dict[str, DataLoader], fold: int, data_dir: str,
        spectrogram_config: Optional[SpectrogramConfig] = None,
        on_epoch_end: Optional[Callable[[int, dict[str, float]], None]] = None,
    ) -> dict[str, float]:
        out_dir = Path(f"{data_dir}/model_trainer/{mlflow.active_run().info.run_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        best_path = out_dir / f"fold{fold}_best.pt"
        best_val_loss = math.inf
        last_eval_best_val_loss = math.inf
        epoch_losses: dict[str, float] = {}
        for epoch in tqdm(range(self.epochs)):
            train_loss = self._run_epoch(loaders["train"], train=True)
            mlflow.log_metric(f"fold{fold}_train_loss", train_loss, step=epoch)
            epoch_losses = {"train_loss": train_loss}
            if self.scheduler is not None:
                mlflow.log_metric(f"fold{fold}_lr", self.optimizer.param_groups[0]["lr"], step=epoch)
                self.scheduler.step()
            if "val" in loaders:
                val_loss = self._run_epoch(loaders["val"], train=False)
                mlflow.log_metric(f"fold{fold}_val_loss", val_loss, step=epoch)
                epoch_losses["val_loss"] = val_loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({"model": self.model, "spectrogram_config": spectrogram_config}, best_path)
            if (
                self.eval_every is not None and (epoch + 1) % self.eval_every == 0
                and epoch < self.epochs - 1 and best_val_loss < last_eval_best_val_loss
            ):
                last_eval_best_val_loss = best_val_loss
                for key, value in self._evaluate(loaders).items():
                    mlflow.log_metric(f"fold{fold}_{key}", value, step=epoch)
            if on_epoch_end is not None and epoch < self.epochs - 1:
                on_epoch_end(epoch, epoch_losses)

        last_path = out_dir / f"fold{fold}_last.pt"
        torch.save({"model": self.model, "spectrogram_config": spectrogram_config}, last_path)
        mlflow.log_artifact(str(last_path))
        if "val" in loaders:
            mlflow.log_metric(f"fold{fold}_best_val_loss", best_val_loss)
            mlflow.log_artifact(str(best_path))
            self.model.load_state_dict(torch.load(best_path, weights_only=False)["model"].state_dict())

        eval_metrics = self._evaluate(loaders)
        for key, value in eval_metrics.items():
            mlflow.log_metric(f"fold{fold}_{key}", value)

        results = {**epoch_losses, **eval_metrics}
        if "val" in loaders:
            results["best_val_loss"] = best_val_loss
        if on_epoch_end is not None:
            on_epoch_end(self.epochs - 1, results)
        return results

    @abstractmethod
    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        """One pass over loader; updates self.model's weights if train,
        otherwise runs forward-only. Returns the sample-weighted mean
        loss."""

    def _evaluate(self, loaders: dict[str, DataLoader]) -> dict[str, float]:
        """Runs once, after fit()'s epoch loop, on self.model (the best
        checkpoint if one was saved). Returns test/val metrics keyed by
        "{split}_{metric_name}"""
        return {}


class SimCLRTrainer(Trainer):
    def __init__(self, config: SimCLRConfig) -> None:
        self.device = torch.device(config.device)
        self.epochs = config.epochs
        self.model = SimCLRModel(config).to(self.device)
        self.augment = SpectrogramSSLAugment(config.augment)
        self.criterion = NTXentLoss(temperature=config.temperature)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.amp = config.amp
        self.max_grad_norm = config.max_grad_norm

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        for specs, _labels in loader:
            specs = specs.to(self.device)
            view0 = self.augment(specs).unsqueeze(1)
            view1 = self.augment(specs).unsqueeze(1)
            with torch.set_grad_enabled(train), self._autocast():
                z0 = self.model(view0)
                z1 = self.model(view1)
                loss = self.criterion(z0, z1)
            if train:
                self._optimizer_step(loss)
            total_loss += loss.item() * specs.size(0)
        return total_loss / len(loader.dataset)


class MoCoTrainer(Trainer):
    def __init__(self, config: MoCoConfig) -> None:
        self.device = torch.device(config.device)
        self.epochs = config.epochs
        self.momentum = config.momentum
        self.model = MoCoModel(config).to(self.device)
        self.augment = SpectrogramSSLAugment(config.augment)
        self.criterion = NTXentLoss(
            temperature=config.temperature,
            memory_bank_size=(config.memory_bank_size, config.projection_out_dim),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.amp = config.amp
        self.max_grad_norm = config.max_grad_norm

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        for specs, _labels in loader:
            specs = specs.to(self.device)
            query_view = self.augment(specs).unsqueeze(1)
            key_view = self.augment(specs).unsqueeze(1)
            with torch.set_grad_enabled(train), self._autocast():
                if train:
                    update_momentum(self.model.backbone, self.model.backbone_momentum, m=self.momentum)
                    update_momentum(self.model.projection_head, self.model.projection_head_momentum, m=self.momentum)
                query = self.model(query_view)
                key = self.model.forward_momentum(key_view)
                loss = self.criterion(query, key)
            if train:
                self._optimizer_step(loss)
            total_loss += loss.item() * specs.size(0)
        return total_loss / len(loader.dataset)


class MoCoV3Trainer(Trainer):
    def __init__(self, config: MoCoV3Config) -> None:
        self.device = torch.device(config.device)
        self.epochs = config.epochs
        self.momentum_base = config.momentum
        self.model = MoCoV3Model(config).to(self.device)
        self.augment = SpectrogramSSLAugment(config.augment)
        self.criterion = NTXentLoss(temperature=config.temperature)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.amp = config.amp
        self.max_grad_norm = config.max_grad_norm
        self._step = 0
        self._total_steps = 0

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        if train and self._total_steps == 0:
            self._total_steps = self.epochs * len(loader)
        total_loss = 0.0
        for specs, _labels in loader:
            specs = specs.to(self.device)
            view0 = self.augment(specs).unsqueeze(1)
            view1 = self.augment(specs).unsqueeze(1)
            with torch.set_grad_enabled(train), self._autocast():
                if train:
                    momentum = cosine_schedule(self._step, self._total_steps, self.momentum_base, 1.0)
                    update_momentum(self.model.backbone, self.model.backbone_momentum, m=momentum)
                    update_momentum(self.model.projection_head, self.model.projection_head_momentum, m=momentum)
                    self._step += 1
                query0, query1 = self.model(view0), self.model(view1)
                key0, key1 = self.model.forward_momentum(view0), self.model.forward_momentum(view1)
                loss = 0.5 * (self.criterion(query0, key1) + self.criterion(query1, key0))
            if train:
                self._optimizer_step(loss)
            total_loss += loss.item() * specs.size(0)
        return total_loss / len(loader.dataset)


class ClassifierTrainer(Trainer):
    def __init__(self, config: ClassifierConfig, num_classes: int) -> None:
        self.device = torch.device(config.device)
        self.epochs = config.epochs
        self.model = ClassifierModel(config, num_classes).to(self.device)
        self.augment = SpectrogramClassifierAugment(config.augment)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = build_optimizer(
            config.optimizer, self.model.parameters(), config.lr, config.weight_decay, config.momentum,
        )
        self.scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs, eta_min=config.lr_scheduler_min_lr,
            )
            if config.lr_scheduler else None
        )
        self.amp = config.amp
        self.max_grad_norm = config.max_grad_norm
        self.eval_every = config.eval_every

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        for specs, labels in loader:
            specs = specs.to(self.device)
            if train:
                specs = self.augment(specs)
            specs, labels = specs.unsqueeze(1), labels.to(self.device)
            with torch.set_grad_enabled(train), self._autocast():
                logits = self.model(specs)
                loss = self.criterion(logits, labels)
            if train:
                self._optimizer_step(loss)
            total_loss += loss.item() * specs.size(0)
        return total_loss / len(loader.dataset)

    def _evaluate(self, loaders: dict[str, DataLoader]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        self.model.eval()
        for split, loader in loaders.items():
            if split == "train":
                continue
            y_true, y_pred, y_score = [], [], []
            with torch.no_grad():
                for specs, labels in loader:
                    specs = specs.to(self.device).unsqueeze(1)
                    probs = torch.softmax(self.model(specs), dim=1)
                    y_true.append(labels.numpy())
                    y_pred.append(probs.argmax(dim=1).cpu().numpy())
                    y_score.append(probs.cpu().numpy())
            y_true_arr, y_pred_arr, y_score_arr = np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_score)
            for name, value in classification_metrics(y_true_arr, y_pred_arr, y_score_arr).items():
                metrics[f"{split}_{name}"] = value
        return metrics


def _mean_metrics(fold_results: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across folds; a key present in only some folds is
    averaged over those folds."""
    keys = {key for result in fold_results for key in result}
    return {key: float(np.mean([r[key] for r in fold_results if key in r])) for key in keys}


def train_model(
    config: ModelTrainerConfig, dataloaders: list[dict[str, DataLoader]], data_dir: str,
    spectrogram_config: Optional[SpectrogramConfig] = None,
    on_epoch_end: Optional[Callable[[int, dict[str, float]], None]] = None,
) -> dict[str, float]:
    """Fits every fold with the paradigm's Trainer; returns its metrics averaged
    across folds. on_epoch_end is forwarded to the classifier Trainer for HPO
    live reporting (the SSL paradigms ignore it)."""
    if config.paradigm == "simclr":
        assert config.simclr is not None, "model_trainer.simclr config is required when paradigm is 'simclr'"
        results = [
            SimCLRTrainer(config.simclr).fit(loaders, fold, data_dir, spectrogram_config)
            for fold, loaders in enumerate(dataloaders)
        ]
    elif config.paradigm == "moco":
        assert config.moco is not None, "model_trainer.moco config is required when paradigm is 'moco'"
        results = [
            MoCoTrainer(config.moco).fit(loaders, fold, data_dir, spectrogram_config)
            for fold, loaders in enumerate(dataloaders)
        ]
    elif config.paradigm == "moco_v3":
        assert config.moco_v3 is not None, "model_trainer.moco_v3 config is required when paradigm is 'moco_v3'"
        results = [
            MoCoV3Trainer(config.moco_v3).fit(loaders, fold, data_dir, spectrogram_config)
            for fold, loaders in enumerate(dataloaders)
        ]
    else:
        assert config.classifier is not None, "model_trainer.classifier config is required when paradigm is 'classifier'"
        num_classes = len(next(iter(dataloaders[0].values())).dataset.dataset.classes)
        results = [
            ClassifierTrainer(config.classifier, num_classes).fit(
                loaders, fold, data_dir, spectrogram_config, on_epoch_end,
            )
            for fold, loaders in enumerate(dataloaders)
        ]
    return _mean_metrics(results)
