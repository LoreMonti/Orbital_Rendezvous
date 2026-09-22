"""Gymnasium environment for the planar rendezvous task.

Observation: the relative state ``[x, y, vx, vy]``, normalised.
Action: the commanded thrust ``[ux, uy]`` in ``[-1, 1]``, scaled by the maximum
thrust. The thrust is continuous, so the agent can correct gently instead of
choosing between a few abrupt burns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from .rewards import RewardConfig


@dataclass(frozen=True)
class EnvConfig:
    """Physical and episode parameters. See ``configs/ppo_default.yaml``."""

    semi_major_axis: float = 6778.0e3
    mu: float = 3.986004418e14
    mass: float = 500.0
    max_thrust: float = 1.0
    time_step: float = 1.0
    max_episode_steps: int = 2000
    initial_radius_range: tuple[float, float] = (80.0, 200.0)
    initial_velocity_scale: float = 0.05
    docking_radius: float = 1.0
    docking_speed: float = 0.05
    max_distance: float = 500.0


class RendezvousEnv(gym.Env):
    """A chaser spacecraft learning to dock with a target in the LVLH frame."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        raise NotImplementedError

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raise NotImplementedError
