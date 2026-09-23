"""The reward function, kept apart from the environment so it can be tuned alone.

Shaping a reward is where most of the design work in this project lives: the
agent optimises exactly what is written here, including the parts we did not
mean to write. The reward of a step is the sum of three terms.

**Shaping** (Ng, Harada & Russell, 1999). With a potential ``Phi(s)``, high
where we want the agent to be, each step adds ``F = gamma Phi(s') - Phi(s)``.
Summed over an episode this telescopes, so it cannot be farmed, and the optimal
policy is provably the same as without it: it speeds learning up without moving
the target. The potential is

    Phi(s) = - w_r r / r_max - w_v max(0, |v| - v_max(r)) / v_ref,
    v_max(r) = v_dock + r / tau.

The first term pulls towards the target. The second is a speed limit that
tightens on approach, a glide slope, ending at the docking speed: without it
the first term alone would teach the agent to rush in and crash.

**Fuel**, ``-w_f |u| dt / m``, the delta-v spent in the step. This one is a real
cost, not shaping: it is meant to change the optimal policy.

**Terminal**: a bonus on docking, a penalty on a crash or a runaway.
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
    """Weights of the individual reward terms.

    ``gamma`` must equal the discount factor of the learning algorithm, or the
    shaping is no longer guaranteed to leave the optimal policy unchanged.
    """

    distance_weight: float = 20.0
    speed_weight: float = 10.0
    fuel_weight: float = 2.0
    approach_time: float = 200.0
    gamma: float = 0.99
    success_bonus: float = 100.0
    failure_penalty: float = -100.0


@dataclass(frozen=True)
class Scales:
    """Physical scales the reward needs from the environment."""

    max_distance: float
    velocity_scale: float
    docking_speed: float
    mass: float
    time_step: float


def speed_limit(distance: float, config: RewardConfig, scales: Scales) -> float:
    """The glide slope ``v_max(r) = v_dock + r / tau``, in m/s."""
    return scales.docking_speed + distance / config.approach_time


def potential(state: np.ndarray, config: RewardConfig, scales: Scales) -> float:
    """Shaping potential: zero at the target at rest, negative everywhere else."""
    distance = float(np.linalg.norm(state[:2]))
    speed = float(np.linalg.norm(state[2:]))
    excess = max(0.0, speed - speed_limit(distance, config, scales))
    return (
        -config.distance_weight * distance / scales.max_distance
        - config.speed_weight * excess / scales.velocity_scale
    )


def step_reward(
    previous_state: np.ndarray,
    state: np.ndarray,
    thrust: np.ndarray,
    terminated: bool,
    config: RewardConfig,
    scales: Scales,
) -> dict[str, float]:
    """Shaping and fuel terms of one step, as a breakdown for logging.

    On a terminated step the potential of the final state is taken as zero:
    the episode has no future to discount, and this is what keeps the shaped
    and the original problem equivalent. A truncated step keeps it, since the
    episode was only interrupted.
    """
    next_potential = 0.0 if terminated else potential(state, config, scales)
    shaping = config.gamma * next_potential - potential(previous_state, config, scales)
    delta_v = float(np.linalg.norm(thrust)) * scales.time_step / scales.mass
    return {"shaping": shaping, "fuel": -config.fuel_weight * delta_v}


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
