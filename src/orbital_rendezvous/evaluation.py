"""Running controllers on a fixed set of starts, and summarising how they did.

Every controller, learned or classical, is flown on the same seeded initial
conditions, so the comparison is on identical problems rather than on
statistics of different ones.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .env import RendezvousEnv
from .rewards import Outcome

# Starts never seen in training, which seeds its environments from 0 upwards.
HELD_OUT_SEED = 10_000

# A controller maps (environment, observation) to an action in [-1, 1]^2: the
# learned policy reads the observation, a classical one the physical state.
Controller = Callable[[RendezvousEnv, np.ndarray], np.ndarray]


@dataclass
class Rollout:
    """One attempt: how it ended, what it cost, and the trajectory for replays."""

    outcome: Outcome
    delta_v: float
    time: float
    final_speed: float
    violated: bool
    positions: np.ndarray = field(repr=False)
    velocities: np.ndarray = field(repr=False)
    thrusts: np.ndarray = field(repr=False)
    start: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class Summary:
    """Statistics over a set of rollouts. Costs are over the docked ones only."""

    success_rate: float
    delta_v_median: float
    delta_v_p5: float
    delta_v_p95: float
    time_median: float
    docking_speed_median: float
    outcomes: dict[str, int]


def rollout(env: RendezvousEnv, controller: Controller, seed: int) -> Rollout:
    """Fly one attempt from the start that ``seed`` selects."""
    obs, _ = env.reset(seed=seed)
    start = env.state.copy()
    positions, velocities, thrusts = [], [], []
    delta_v, done, violated = 0.0, False, False
    while not done:
        obs, _, terminated, truncated, info = env.step(controller(env, obs))
        delta_v += info["delta_v"]
        violated = violated or info.get("keep_out_violated", False)
        positions.append(info["position"])
        velocities.append(info["velocity"])
        thrusts.append(info["thrust"])
        done = terminated or truncated
    return Rollout(
        outcome=info["outcome"],
        delta_v=delta_v,
        time=env.steps * env.config.time_step,
        final_speed=info["speed"],
        violated=violated,
        positions=np.array(positions),
        velocities=np.array(velocities),
        thrusts=np.array(thrusts),
        start=start,
    )


def evaluate(env: RendezvousEnv, controller: Controller, seeds: Sequence[int]) -> list[Rollout]:
    return [rollout(env, controller, seed) for seed in seeds]


def summarise(rollouts: Sequence[Rollout]) -> Summary:
    docked = [r for r in rollouts if r.outcome is Outcome.DOCKED]
    outcomes: dict[str, int] = {}
    for r in rollouts:
        outcomes[r.outcome.value] = outcomes.get(r.outcome.value, 0) + 1

    def stat(values, fn):
        return float(fn(values)) if values else float("nan")

    dv = [r.delta_v for r in docked]
    return Summary(
        success_rate=len(docked) / len(rollouts),
        delta_v_median=stat(dv, np.median),
        delta_v_p5=stat(dv, lambda x: np.percentile(x, 5)),
        delta_v_p95=stat(dv, lambda x: np.percentile(x, 95)),
        time_median=stat([r.time for r in docked], np.median),
        docking_speed_median=stat([r.final_speed for r in docked], np.median),
        outcomes=outcomes,
    )


def start_angle(position: np.ndarray) -> float:
    """Angle between a position and the docking axis +y, in degrees, from 0 to 180."""
    return float(np.degrees(np.arctan2(abs(position[0]), position[1])))


def docked_by_sector(
    rollouts: Sequence[Rollout], edges: Sequence[float] = (0.0, 45.0, 90.0, 135.0, 180.0)
) -> list[tuple[int, int]]:
    """Dockings and attempts for each sector of start angles from the docking axis.

    With an oriented target, the direction a start comes from sets how hard
    it is: from behind the station the chaser must go around the keep-out
    sphere. The last sector includes 180 degrees.
    """
    angles = np.array([start_angle(r.start) for r in rollouts])
    docked = np.array([r.outcome is Outcome.DOCKED for r in rollouts])
    counts = []
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        last = i == len(edges) - 2
        inside = (angles >= low) & ((angles <= high) if last else (angles < high))
        counts.append((int(docked[inside].sum()), int(inside.sum())))
    return counts


def pareto_front(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of the points not dominated in both coordinates, lower being better.

    Returned in increasing order of the first coordinate.
    """
    order = sorted(range(len(points)), key=lambda i: points[i])
    front, best_second = [], np.inf
    for i in order:
        if points[i][1] < best_second:
            front.append(i)
            best_second = points[i][1]
    return front
