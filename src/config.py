"""Shared config loading helpers for command-line dataset tools."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_yaml_config(path: Path, model: type[ConfigT]) -> ConfigT:
    """Load a YAML file and validate it with a Pydantic model."""

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return model.model_validate(data)
