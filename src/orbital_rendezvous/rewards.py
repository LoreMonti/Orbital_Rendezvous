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

With a keep-out sphere and an approach cone (see `env`), the distance ``r`` in
the first term becomes the length of the shortest allowed path: straight in
from inside the cone; otherwise around the sphere, along a tangent and an arc
of its rim, to the mouth of the cone, then in along it. The cone is the one of
the moment: `callbacks.ConeCurriculum` starts it at 180 degrees, where every
point is inside it and the potential is the plain straight distance, and
narrows it as the agent masters it. The potential and the constraint change
together. Imposed at the final cone from the start, the same potential stopped
the agent from ever learning to dock.

**Fuel**, ``-w_f |u| dt / m``, the delta-v spent in the step. This one is a real
cost, not shaping: it is meant to change the optimal policy.

**Terminal**: a bonus on docking, a penalty on a crash, a runaway or a
keep-out violation.
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
    KEEP_OUT = "keep-out violation"


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
    keep_out_radius: float = 0.0
    approach_cone_deg: float = 15.0


def in_approach_cone(position: np.ndarray, cone_deg: float) -> bool:
    """Whether ``(x, y)`` lies inside the cone of half-angle ``cone_deg`` around +y."""
    distance = float(np.hypot(*position))
    return distance == 0.0 or position[1] >= distance * np.cos(np.radians(cone_deg))


def _segment_clearance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance from the origin to the segment ``a -> b``."""
    d = b - a
    t = float(np.clip(-(a @ d) / max(float(d @ d), 1e-12), 0.0, 1.0))
    return float(np.hypot(*(a + t * d)))


def _around_disc(position: np.ndarray, rim_point: np.ndarray, radius: float) -> float:
    """Shortest path from ``position`` to ``rim_point`` that stays out of the disc."""
    distance = float(np.hypot(*position))
    straight = float(np.hypot(*(position - rim_point)))
    if distance <= radius or _segment_clearance(position, rim_point) >= radius * (1 - 1e-9):
        return straight
    # A tangent to the rim, then along the rim: angles measured from +y.
    tangent = np.sqrt(distance**2 - radius**2)
    half = np.arccos(radius / distance)
    here = np.arctan2(position[0], position[1])
    there = np.arctan2(rim_point[0], rim_point[1])
    arcs = []
    for touch in (here + half, here - half):
        arcs.append(abs((there - touch + np.pi) % (2 * np.pi) - np.pi))
    return float(tangent + radius * min(arcs))


def path_length(position: np.ndarray, scales: Scales) -> float:
    """Length of the shortest path to the target that respects the keep-out sphere.

    From inside the approach cone, or without a sphere, the straight distance.
    Otherwise around the sphere to the nearer edge of the cone's mouth, the arc
    of the rim inside the cone, then along the cone to the target:
    ``sqrt(r^2 - R^2) + R dphi + R`` when the rim is in the way.
    """
    radius, cone = scales.keep_out_radius, scales.approach_cone_deg
    if not radius or in_approach_cone(position, cone):
        return float(np.hypot(*position))
    edges = [radius * np.array([np.sin(a), np.cos(a)]) for a in np.radians([cone, -cone])]
    return min(_around_disc(position, edge, radius) for edge in edges) + radius


def speed_limit(distance: float, config: RewardConfig, scales: Scales) -> float:
    """The glide slope ``v_max(r) = v_dock + r / tau``, in m/s."""
    return scales.docking_speed + distance / config.approach_time


def potential(state: np.ndarray, config: RewardConfig, scales: Scales) -> float:
    """Shaping potential: zero at the target at rest, negative everywhere else."""
    distance = float(np.linalg.norm(state[:2]))
    speed = float(np.linalg.norm(state[2:]))
    excess = max(0.0, speed - speed_limit(distance, config, scales))
    return (
        -config.distance_weight * path_length(state[:2], scales) / scales.max_distance
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
    if outcome in (Outcome.CRASHED, Outcome.ESCAPED, Outcome.KEEP_OUT):
        return config.failure_penalty
    return 0.0
