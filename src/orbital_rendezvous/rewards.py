"""The reward function, kept apart from the environment so it can be tuned alone.

Shaping a reward is where most of the design work in this project lives: the
agent optimises exactly what is written here, including the parts we did not
mean to write.

For now only the terminal reward is in place: a bonus on docking and a penalty
on a crash or a runaway. The per-step shaping terms come next.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Outcome(str, Enum):
    """How an episode ended. ``None`` in the environment means it is still running."""

    DOCKED = "docked"
    CRASHED = "crashed"
    ESCAPED = "escaped"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class RewardConfig:
    """Weights of the individual reward terms."""

    distance_weight: float = 1e-2
    fuel_weight: float = 1e-3
    success_bonus: float = 100.0
    failure_penalty: float = -100.0


def step_reward(
    state: np.ndarray,
    thrust: np.ndarray,
    previous_state: np.ndarray,
    config: RewardConfig,
) -> tuple[float, dict[str, float]]:
    """Reward for a single step, plus a breakdown of its terms for logging."""
    raise NotImplementedError


def terminal_reward(outcome: Outcome | None, config: RewardConfig) -> float:
    """Bonus on docking, penalty on a crash or a runaway, zero otherwise.

    A timeout is not penalised: it is a truncation, not a failure, and the
    agent should not learn to fear the clock itself.
    """
    if outcome is Outcome.DOCKED:
        return config.success_bonus
    if outcome in (Outcome.CRASHED, Outcome.ESCAPED):
        return config.failure_penalty
    return 0.0
