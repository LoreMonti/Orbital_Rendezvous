"""Configuration loading: from the YAML file to the dataclasses the code uses."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from .env import EnvConfig
from .rewards import RewardConfig


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML configuration file into a plain dictionary."""
    with open(path) as handle:
        return yaml.safe_load(handle)


def _build(cls, values: dict[str, Any], section: str):
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        # A typo in the YAML would otherwise be silently replaced by a default.
        raise ValueError(f"unknown keys in '{section}': {sorted(unknown)}")
    return cls(**values)


def build_configs(config: dict[str, Any]) -> tuple[EnvConfig, RewardConfig]:
    """Environment and reward configurations from the ``environment`` and ``rewards`` sections.

    The shaping discount must equal the discount of the learning algorithm, or
    the shaping is no longer guaranteed to leave the optimal policy unchanged,
    so a mismatch is an error rather than a silent choice between the two.
    """
    env_values = dict(config["environment"])
    for key in ("initial_radius_range", "start_angle_range_deg"):
        if key in env_values:
            env_values[key] = tuple(env_values[key])
    env_config = _build(EnvConfig, env_values, "environment")
    reward_config = _build(RewardConfig, dict(config["rewards"]), "rewards")

    training_gamma = config.get("training", {}).get("gamma")
    if training_gamma is not None and training_gamma != reward_config.gamma:
        raise ValueError(
            f"rewards.gamma ({reward_config.gamma}) must equal training.gamma ({training_gamma})"
        )
    return env_config, reward_config
