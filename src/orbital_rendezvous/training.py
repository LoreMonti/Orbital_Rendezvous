"""Building and training the PPO agent from a configuration dictionary.

Shared by `scripts/train.py` and `scripts/fuel_study.py`, so that a run of the
study is, by construction, the same training as a normal run with a different
configuration.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed

from .env import RendezvousEnv
from .utils import build_configs

PPO_KEYS = (
    "learning_rate", "n_steps", "batch_size", "gamma", "gae_lambda", "clip_range", "ent_coef",
)


def with_overrides(
    config: dict[str, Any],
    gamma: float | None = None,
    fuel_weight: float | None = None,
    curriculum: dict[str, float] | None = None,
    thrust_deadzone: float | None = None,
) -> dict[str, Any]:
    """A copy of ``config`` with a new discount, fuel weight, curriculum and deadzone.

    The discount is set in both places it lives, the PPO settings and the
    reward shaping, since they must agree for the shaping to leave the optimal
    policy unchanged. With a curriculum, ``fuel_weight`` is the final weight.
    """
    config = copy.deepcopy(config)
    if gamma is not None:
        config["training"]["gamma"] = gamma
        config["rewards"]["gamma"] = gamma
    if fuel_weight is not None:
        config["rewards"]["fuel_weight"] = fuel_weight
    if curriculum is not None:
        config["training"]["fuel_curriculum"] = curriculum
    if thrust_deadzone is not None:
        config["environment"]["thrust_deadzone"] = thrust_deadzone
    return config


def build_model(config: dict[str, Any], seed: int, run_dir: Path) -> PPO:
    """PPO on parallel environments, logging to ``run_dir``."""
    env_config, reward_config = build_configs(config)
    training = config["training"]
    if training.get("algorithm", "PPO") != "PPO":
        raise ValueError("only PPO is supported")

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    set_random_seed(seed)
    vec_env = make_vec_env(
        lambda: RendezvousEnv(env_config, reward_config),
        n_envs=training["n_envs"],
        seed=seed,
        monitor_dir=str(run_dir / "monitor"),
        monitor_kwargs={"info_keywords": ("is_success",)},
    )
    model = PPO(
        training["policy"],
        vec_env,
        policy_kwargs=training.get("policy_kwargs"),
        seed=seed,
        verbose=0,
        **{key: training[key] for key in PPO_KEYS},
    )
    model.set_logger(configure(str(run_dir), ["csv"]))
    return model


def train(
    config: dict[str, Any],
    seed: int,
    total_timesteps: int,
    run_dir: Path,
    output: Path,
    callbacks: list[BaseCallback] | None = None,
    checkpoints: bool = True,
) -> PPO:
    """Train, save the model to ``output`` and return it."""
    model = build_model(config, seed, run_dir)
    curriculum = config["training"].get("fuel_curriculum")
    if curriculum:
        from .callbacks import FuelCurriculum

        # The fuel weight of the rewards section is where the schedule ends:
        # the task the agent is finally trained and judged on.
        callbacks = [*(callbacks or []), FuelCurriculum(
            curriculum["start"], config["rewards"]["fuel_weight"], curriculum["ramp"]
        )]
    every = [
        CheckpointCallback(
            save_freq=max(1, 200_000 // config["training"]["n_envs"]),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
        )
    ] if checkpoints else []
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(every + (callbacks or [])))
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    return model
