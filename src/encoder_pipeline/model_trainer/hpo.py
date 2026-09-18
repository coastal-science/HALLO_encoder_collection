"""Ray Tune hyperparameter search over the classifier paradigm, driven by
model_trainer.tune. Each trial is an MLflow run nested under the trainer run."""

from pathlib import Path
from typing import Any, Optional

import mlflow
from mlflow.entities import RunStatus
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from encoder_pipeline.common.config_utils import deep_merge
from encoder_pipeline.common.mlflow_utils import flatten_params
from encoder_pipeline.model_trainer.config import ModelTrainerConfig, RayTuneConfig, SearchParam
from encoder_pipeline.model_trainer.data_loader import build_dataloaders
from encoder_pipeline.model_trainer.train import train_model
from encoder_pipeline.preprocessor.config import SpectrogramConfig


def build_search_space(search_space: dict[str, SearchParam]) -> dict[str, Any]:
    """Translates each SearchParam into its ray.tune sampler, keyed by the same
    dotted config path."""
    space: dict[str, Any] = {}
    for param_name, param in search_space.items():
        (domain, value), = ((name, val) for name, val in param.model_dump().items() if val is not None)
        sampler = getattr(tune, domain)
        space[param_name] = sampler(value) if domain in ("choice", "grid_search") else sampler(*value)
    return space


def apply_sample(base: ModelTrainerConfig, sample: dict[str, Any]) -> ModelTrainerConfig:
    """Deep-merges a flat {dotted.path: value} Tune sample onto base and
    re-validates, so a mistyped path is rejected by the strict config."""
    overrides: dict = {}
    for path, value in sample.items():
        node = overrides
        *parents, leaf = path.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return ModelTrainerConfig.model_validate(deep_merge(base.model_dump(), overrides))


def _finalize_trial_runs(result_grid: tune.ResultGrid, driver_run_id: str, tracking_uri: str) -> None:
    """Close out child runs that Ray killed mid-report during teardown (best /
    last trials block in tune.report after their final result and never reach
    the trainable's own end_run), so the UI stops showing them as running.
    Trials Ray recorded an error for are marked FAILED, the rest FINISHED."""
    client = mlflow.tracking.MlflowClient(tracking_uri)
    experiment_id = client.get_run(driver_run_id).info.experiment_id
    errored = {
        result.metrics.get("trial_id")
        for result in result_grid
        if result.error is not None and result.metrics
    }
    for run in client.search_runs(
        [experiment_id],
        filter_string=f"tags.`{MLFLOW_PARENT_RUN_ID}` = '{driver_run_id}'",
    ):
        if run.info.status != "RUNNING":
            continue
        failed = any(trial_id and trial_id in run.info.run_name for trial_id in errored)
        client.set_terminated(run.info.run_id, "FAILED" if failed else "FINISHED")


def run_tuning(
    config: ModelTrainerConfig,
    dataset_path: str,
    data_dir: str,
    driver_run_id: str,
    tracking_uri: str,
    experiment_name: str,
    spectrogram_config: Optional[SpectrogramConfig] = None,
) -> dict[str, Any]:
    """Runs model_trainer.tune's search and logs the best config / metric to the
    driver run. Returns {"best_config", "best_metrics"}."""
    tune_cfg = config.tune
    assert config.classifier is not None, "model_trainer.classifier is required when tune is enabled"
    assert config.dataloader.n_folds == 1, "model_trainer.tune expects dataloader.n_folds == 1"

    dataset_path = str(Path(dataset_path).resolve())
    data_dir = str(Path(data_dir).resolve())

    base = config.model_copy(update={"tune": RayTuneConfig()}, deep=True)
    if base.dataloader.splits_path:
        base.dataloader.splits_path = str(Path(base.dataloader.splits_path).resolve())

    def trainable(sample: dict[str, Any]) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        trial_config = apply_sample(base, sample)
        mlflow.start_run(
            run_name=tune.get_context().get_trial_name(),
            tags={MLFLOW_PARENT_RUN_ID: driver_run_id},
        )
        try:
            mlflow.log_params(flatten_params("model_trainer", trial_config.model_dump()))
            mlflow.log_param("dataset_path", dataset_path)
            dataloaders = build_dataloaders(dataset_path, trial_config.dataloader, data_dir)
            train_model(
                trial_config, dataloaders, data_dir, spectrogram_config,
                on_epoch_end=lambda _epoch, metrics: tune.report(metrics),
            )
        except SystemExit:
            # ASHA stops a trial by raising SystemExit(0) from inside tune.report;
            # an early stop is a normal outcome, so close the run as FINISHED.
            mlflow.end_run(RunStatus.to_string(RunStatus.FINISHED))
            raise
        except BaseException:
            mlflow.end_run(RunStatus.to_string(RunStatus.FAILED))
            raise
        else:
            mlflow.end_run(RunStatus.to_string(RunStatus.FINISHED))

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=config.classifier.epochs,
        grace_period=tune_cfg.grace_period,
        reduction_factor=tune_cfg.reduction_factor,
    )
    storage_path = tune_cfg.storage_path or str(Path(data_dir).resolve() / "model_trainer" / "ray_tune")

    tuner = tune.Tuner(
        tune.with_resources(trainable, tune_cfg.resources_per_trial),
        param_space=build_search_space(tune_cfg.search_space),
        tune_config=tune.TuneConfig(
            metric=tune_cfg.metric,
            mode=tune_cfg.mode,
            num_samples=tune_cfg.num_samples,
            max_concurrent_trials=tune_cfg.max_concurrent_trials,
            scheduler=scheduler,
        ),
        run_config=tune.RunConfig(name=f"tune_{driver_run_id}", storage_path=storage_path),
    )

    result_grid = tuner.fit()
    _finalize_trial_runs(result_grid, driver_run_id, tracking_uri)
    best = result_grid.get_best_result(tune_cfg.metric, tune_cfg.mode)
    mlflow.log_params({f"best.{path}": value for path, value in (best.config or {}).items()})
    if tune_cfg.metric in best.metrics:
        mlflow.log_metric(f"best_{tune_cfg.metric}", best.metrics[tune_cfg.metric])
    return {"best_config": best.config, "best_metrics": best.metrics}
