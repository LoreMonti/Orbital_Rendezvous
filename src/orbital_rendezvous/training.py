"""Building and training the PPO agent from a configuration dictionary.

Shared by `scripts/train.py` and `scripts/fuel_study.py`, so that a run of the
study is, by construction, the same training as a normal run with a different
configuration.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed

from .env import RendezvousEnv
from .utils import build_configs, make_env

PPO_KEYS = (
    "learning_rate", "n_steps", "batch_size", "gamma", "gae_lambda", "clip_range", "ent_coef",
)


def with_overrides(
    config: dict[str, Any],
    gamma: float | None = None,
    fuel_weight: float | None = None,
    curriculum: dict[str, float] | None = None,
    thrust_deadzone: float | None = None,
    engine_switch: bool | None = None,
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
    if engine_switch is not None:
        config["environment"]["engine_switch"] = engine_switch
    return config


def build_model(config: dict[str, Any], seed: int, run_dir: Path) -> PPO:
    """PPO on parallel environments, logging to ``run_dir``."""
    build_configs(config)   # validates the configuration before anything is written
    training = config["training"]
    if training.get("algorithm", "PPO") != "PPO":
        raise ValueError("only PPO is supported")

    run_dir.mkdir(parents=True, exist_ok=True)
    # The seed actually used, which a command-line override may have changed.
    saved = copy.deepcopy(config)
    saved["training"]["seed"] = seed
    with open(run_dir / "config.yaml", "w") as handle:
        yaml.safe_dump(saved, handle, sort_keys=False)

    set_random_seed(seed)
    vec_env = make_vec_env(
        lambda: make_env(config),
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
    """Train, save the model to ``output`` and return it.

    With ``training.start_curriculum`` the history of its stages is also saved
    to ``curriculum.json`` in the run directory. With ``training.best_model``
    the best policy on validation starts is saved next to ``output`` as
    ``<name>_best.zip``, and its measurements to ``best_model.json``.
    """
    starts = config["training"].get("start_curriculum")
    stages: dict[str, Any] = {}
    if starts:
        from .callbacks import ConeCurriculum, StartCurriculum

        env_config, reward_config = build_configs(config)
        front = replace(env_config, start_angle_range_deg=(0.0, starts["start_deg"]))
        rule = dict(step_deg=starts["step_deg"], success_threshold=starts["threshold"])
        if starts.get("cone_start_deg") is not None:
            # Phase 1: learn to dock, then narrow the cone, from in front of the port.
            stages["cone"] = ConeCurriculum(
                lambda: RendezvousEnv(front, reward_config),
                final_deg=env_config.approach_cone_deg,
                start_deg=starts["cone_start_deg"],
                **rule,
            )
        # Phase 2: the final cone, starts widened towards the back of the station.
        stages["starts"] = StartCurriculum(
            lambda: RendezvousEnv(env_config, reward_config),
            start_deg=starts["start_deg"],
            final_deg=starts.get("final_deg", 180.0),
            after=stages.get("cone"),
            **rule,
        )
        callbacks = [*(callbacks or []), *stages.values()]
    model = build_model(config, seed, run_dir)
    if stages:
        # Before the first reset, which `learn` does before any callback runs,
        # so that the first episodes too start in the first stage.
        model.get_env().env_method("set_start_angles", 0.0, starts["start_deg"])
    curriculum = config["training"].get("fuel_curriculum")
    if curriculum:
        from .callbacks import FuelCurriculum

        # The fuel weight of the rewards section is where the schedule ends:
        # the task the agent is finally trained and judged on.
        callbacks = [*(callbacks or []), FuelCurriculum(
            curriculum["start"], config["rewards"]["fuel_weight"], curriculum["ramp"]
        )]
    best = None
    if config["training"].get("best_model"):
        from .callbacks import BestModel

        # Saved next to the final model: models/x.zip and models/x_best.zip.
        best = BestModel(lambda: make_env(config), output.with_name(output.stem + "_best.zip"),
                         **config["training"]["best_model"])
        callbacks = [*(callbacks or []), best]
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
    if stages:
        with open(run_dir / "curriculum.json", "w") as handle:
            json.dump({name: c.history for name, c in stages.items()}, handle, indent=1)
    if best is not None:
        with open(run_dir / "best_model.json", "w") as handle:
            json.dump(best.history, handle, indent=1)
    return model
