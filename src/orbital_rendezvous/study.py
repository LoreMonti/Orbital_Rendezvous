"""The fuel study: how much of the agent's hurry comes from the discount.

With a discount ``gamma`` a docking bonus earned after ``K`` steps is worth
``100 gamma^K``: at ``gamma = 0.99`` docking after 60 steps is worth 55, after
288 steps only 5.5. Hurrying is worth far more than the fuel it costs. The study
trains the agent over a grid of discounts and fuel weights, several seeds each,
and places every run on the same delta-v against time plot as the LQR sweep.

A fuel weight above about 2 from the very first step makes moving cost more
than approaching earns before the docking bonus has ever been seen, and the
agent learns to stay put, or not, depending on the seed. Every run therefore
starts from a weight of 2 and raises it to its target over the first part of
training (see `callbacks.FuelCurriculum`): learn to dock first, then to save.

The fuel cost is paid on the thrust actually commanded, exploration noise
included, so during training a slow approach also pays for the noise of every
step spent waiting, and looks expensive. A minimum thruster level (the
environment's ``thrust_deadzone``) switches the engine off for commands near
zero, so that coasting is free.

One job trains and evaluates one (gamma, fuel weight, seed) triple. Jobs are
independent and run in separate processes; each writes its result to disk, so
an interrupted study resumes where it stopped.
"""

from __future__ import annotations

import json
import shutil
import signal
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .env import RendezvousEnv
from .evaluation import HELD_OUT_SEED, evaluate, summarise
from .training import train, with_overrides
from .utils import build_configs


@dataclass(frozen=True)
class Job:
    """One training run of the study."""

    gamma: float
    fuel_weight: float
    seed: int
    timesteps: int
    episodes: int
    directory: str
    curriculum_start: float = 2.0
    curriculum_ramp: float = 0.5
    thrust_deadzone: float = 0.0

    @property
    def label(self) -> str:
        deadzone = f"-deadzone{self.thrust_deadzone:g}" if self.thrust_deadzone else ""
        return f"gamma{self.gamma:g}-fuel{self.fuel_weight:g}{deadzone}-seed{self.seed}"

    @property
    def result_path(self) -> Path:
        return Path(self.directory) / self.label / "result.json"


def make_jobs(
    gammas: list[float],
    fuel_weights: list[float],
    seeds: list[int],
    timesteps: int,
    episodes: int,
    directory: str,
    curriculum_start: float = 2.0,
    curriculum_ramp: float = 0.5,
    thrust_deadzone: float = 0.0,
) -> list[Job]:
    """Every (gamma, fuel weight, seed) combination, slowest-learning first.

    The largest discounts learn slowest, so they are started first and do not
    end up running alone at the end of the study.
    """
    return [
        Job(g, w, s, timesteps, episodes, directory, curriculum_start, curriculum_ramp,
            thrust_deadzone)
        for g in sorted(gammas, reverse=True)
        for w in fuel_weights
        for s in seeds
    ]


def ignore_interrupts() -> None:
    """Worker initialiser: leave Ctrl-C to the parent, which stops every worker at once."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_job(job: Job, base_config: dict[str, Any]) -> dict[str, Any]:
    """Train one configuration, evaluate it on the held-out starts, save the result."""
    import torch
    from stable_baselines3 import PPO

    # Several jobs run side by side: one thread each avoids oversubscribing cores.
    torch.set_num_threads(1)
    curriculum = {"start": job.curriculum_start, "ramp": job.curriculum_ramp}
    config = with_overrides(
        base_config, gamma=job.gamma, fuel_weight=job.fuel_weight, curriculum=curriculum,
        thrust_deadzone=job.thrust_deadzone,
    )
    run_dir = job.result_path.parent
    if run_dir.exists():
        # Left half-way by an interrupted study: start this run again from scratch.
        shutil.rmtree(run_dir)
    model_path = run_dir / "model.zip"
    model: PPO = train(config, job.seed, job.timesteps, run_dir, model_path, checkpoints=False)

    env_config, reward_config = build_configs(config)
    env = RendezvousEnv(env_config, reward_config)
    seeds = range(HELD_OUT_SEED, HELD_OUT_SEED + job.episodes)
    runs = evaluate(env, lambda e, obs: model.predict(obs, deterministic=True)[0], seeds)
    result = {**asdict(job), "summary": asdict(summarise(runs))}
    job.result_path.write_text(json.dumps(result, indent=2))
    return result


RELIABLE = 0.95
"""Docking rate a run needs for its costs to count. A run that docks rarely docks
only from the easy starts, and its median delta-v would flatter it."""


def aggregate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (gamma, fuel weight): medians over the seeds that dock reliably.

    Costs are taken only over reliable runs; how many seeds were reliable is
    reported alongside, since a configuration that fails on some seeds is not
    a solution even if its successful seeds are cheap.
    """
    groups: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault((result["gamma"], result["fuel_weight"]), []).append(result)
    rows = []
    for (gamma, weight), group in sorted(groups.items()):
        group = sorted(group, key=lambda r: r["seed"])
        reliable = [r["summary"] for r in group if r["summary"]["success_rate"] >= RELIABLE]

        def median(key, summaries=reliable):
            values = [s[key] for s in summaries]
            return float(np.median(values)) if values else float("nan")

        rows.append({
            "gamma": gamma,
            "fuel_weight": weight,
            "seeds": len(group),
            "reliable_seeds": len(reliable),
            "success_rates": [r["summary"]["success_rate"] for r in group],
            "delta_v_median": median("delta_v_median"),
            "time_median": median("time_median"),
            "docking_speed_median": median("docking_speed_median"),
        })
    return rows
