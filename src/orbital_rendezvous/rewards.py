"""The reward function, kept apart from the environment so it can be tuned alone.

Shaping a reward is where most of the design work in this project lives: the
agent optimises exactly what is written here, including the parts we did not
mean to write.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def terminal_reward(docked: bool, config: RewardConfig) -> float:
    """Bonus on a successful docking, penalty on a crash or a runaway."""
    raise NotImplementedError
