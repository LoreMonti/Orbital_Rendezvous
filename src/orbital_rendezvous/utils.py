"""Configuration loading: from the YAML file to the dataclasses the code uses."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from .env import EnvConfig, GoToConfig, GoToEnv, RendezvousEnv
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


def build_goal_config(config: dict[str, Any]) -> GoToConfig | None:
    """The ``goal`` section, present only for the pilot that flies to a point."""
    if "goal" not in config:
        return None
    values = dict(config["goal"])
    if "goals" in values:
        values["goals"] = tuple(tuple(float(c) for c in g) for g in values["goals"])
    if "menu_radii" in values:
        values["menu_radii"] = tuple(float(r) for r in values["menu_radii"])
    return _build(GoToConfig, values, "goal")


def load_pilot(planner: dict[str, Any]):
    """The two frozen pilots of a ``planner`` section, as a `hierarchy.HierarchicalPilot`."""
    from stable_baselines3 import PPO

    from .hierarchy import HierarchicalPilot

    configs = [build_configs(load_config(planner[key]))[0]
               for key in ("go_to_config", "final_config")]
    return HierarchicalPilot.from_models(
        PPO.load(planner["go_to"], device="cpu"), PPO.load(planner["final"], device="cpu"),
        *configs,
    )


def make_env(config: dict[str, Any]):
    """The environment a configuration describes.

    `GoToEnv` with a ``goal`` section; `hierarchy.PlannerEnv`, choosing the
    waypoint that two frozen pilots then fly, with a ``planner`` section; the
    plain `RendezvousEnv` otherwise.
    """
    env_config, reward_config = build_configs(config)
    if "planner" in config:
        from .env import waypoint_menu
        from .hierarchy import PlannerEnv

        planner = config["planner"]
        pilot = load_pilot(planner)
        pilot.keep_out = env_config.keep_out_radius
        menu = waypoint_menu(tuple(planner["menu_radii"]), planner["menu_directions"])
        return PlannerEnv(RendezvousEnv(env_config, reward_config), pilot, menu,
                          planner["fuel_weight"])
    goal_config = build_goal_config(config)
    if goal_config is not None:
        return GoToEnv(env_config, reward_config, goal_config)
    return RendezvousEnv(env_config, reward_config)
