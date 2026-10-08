"""The go-to task: fly to a goal point and stop there, for the pilot between waypoints.

Moved out of the package with the rest of the teacher-student study (Steps
16-20); it subclasses `orbital_rendezvous.core.env.RendezvousEnv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from orbital_rendezvous.core.env import EnvConfig, RendezvousEnv
from orbital_rendezvous.core.rewards import Outcome, RewardConfig


@dataclass(frozen=True)
class GoToConfig:
    """Goals and starts of `GoToEnv`. See ``configs/ppo_goto.yaml``.

    Goals are the points the planner of `hierarchy.HierarchicalPilot` flies
    to, each moved at random by up to ``goal_jitter`` so that the agent learns
    to reach a point, not to memorise three. A fraction of the starts is set
    up as the planner's handover at a waypoint: within ``moving_start_offset``
    of one, already moving at up to ``moving_start_speed``.
    """

    goals: tuple[tuple[float, float], ...] = ((0.0, 30.0), (50.0, 0.0), (-50.0, 0.0))
    goal_jitter: float = 5.0
    moving_start_fraction: float = 0.3
    moving_start_offset: float = 5.0
    moving_start_speed: float = 0.3
    # The menu of waypoints a learned planner chooses from (`waypoint_menu`),
    # added to ``goals``: ``menu_directions`` around the station at each of
    # ``menu_radii``. Empty for the pilot of Step 16.
    menu_radii: tuple[float, ...] = ()
    menu_directions: int = 16

    def all_goals(self) -> np.ndarray:
        """The goal points: ``goals``, then the menu."""
        points = [np.asarray(self.goals, dtype=float).reshape(-1, 2)]
        if self.menu_radii:
            points.append(waypoint_menu(self.menu_radii, self.menu_directions))
        return np.vstack(points)


def waypoint_menu(radii: tuple[float, ...], directions: int) -> np.ndarray:
    """Waypoints around the station: ``directions`` evenly spaced angles at each radius.

    Angles are measured from the docking axis +y, the first on it; rows run
    through the angles of the first radius, then of the next.
    """
    angles = np.radians(np.arange(directions) * 360.0 / directions)
    return np.array([[r * np.sin(a), r * np.cos(a)] for r in radii for a in angles])


def goto_observation(
    state: np.ndarray, goal: np.ndarray, elapsed: float, config: EnvConfig
) -> np.ndarray:
    """What `GoToEnv` shows the agent: where it is from the goal, and where in absolute terms.

    ``[(x - gx) / r_max, (y - gy) / r_max, vx / v_ref, vy / v_ref, gx / r_max, gy / r_max,
    t / T_max]``. The relative position says where to go; the absolute one is
    needed too, since the Clohessy-Wiltshire equations are not invariant under
    a shift: at rest at ``x``, the chaser needs a thrust ``-3 n^2 x m`` to stay.
    """
    r, v = config.max_distance, config.velocity_scale
    return np.array([
        (state[0] - goal[0]) / r, (state[1] - goal[1]) / r, state[2] / v, state[3] / v,
        goal[0] / r, goal[1] / r, elapsed,
    ], dtype=np.float32)


class GoToEnv(RendezvousEnv):
    """Fly to a goal point and stop there: the pilot between the waypoints of a plan.

    The task of the default agent with the target moved from the origin to a
    goal ``g``. Reaching it, the counterpart of docking, means ending a step
    within ``docking_radius`` of it and slower than ``docking_speed``; the
    shaping potential and its glide slope measure distance from ``g``. Passing
    through the goal too fast is not a failure, only not yet a success. There
    is no keep-out sphere and no collision with the station: keeping the
    chaser clear of it is the planner's job, which places its waypoints so
    that the way between them stays clear.
    """

    def __init__(
        self,
        config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
        goal_config: GoToConfig | None = None,
    ) -> None:
        super().__init__(config, reward_config)
        self.goal_config = goal_config or GoToConfig()
        low = np.array([-4.0, -4.0, -20.0, -20.0, -1.0, -1.0, 0.0], dtype=np.float32)
        high = np.array([4.0, 4.0, 20.0, 20.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def _observation(self) -> np.ndarray:
        obs = goto_observation(
            self.state, self.goal, self.steps / self.config.max_episode_steps, self.config
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _info(self, outcome: Outcome | None, thrust: np.ndarray) -> dict[str, Any]:
        info = super()._info(outcome, thrust)
        info["goal"] = self.goal.copy()
        info["goal_distance"] = float(np.hypot(*(self.state[:2] - self.goal)))
        return info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        cfg, rng = self.goal_config, self.np_random
        goals = cfg.all_goals()
        self.goal = goals[rng.integers(len(goals))].copy()
        self.goal += rng.uniform(-cfg.goal_jitter, cfg.goal_jitter, size=2)
        if rng.random() < cfg.moving_start_fraction:
            # A handover at a waypoint: near one, still moving.
            waypoint = goals[rng.integers(len(goals))]
            offset = rng.uniform(-cfg.moving_start_offset, cfg.moving_start_offset, size=2)
            heading = rng.uniform(0.0, 2.0 * np.pi)
            speed = rng.uniform(0.0, cfg.moving_start_speed)
            self.state = np.append(waypoint + offset, speed * np.array(
                [np.cos(heading), np.sin(heading)]))
        return self._observation(), self._info(None, np.zeros(2))

    def _outcome(self, previous_position: np.ndarray, violated: bool) -> Outcome | None:
        cfg = self.config
        near = float(np.hypot(*(self.state[:2] - self.goal))) < cfg.docking_radius
        if near and float(np.hypot(*self.state[2:])) < cfg.docking_speed:
            return Outcome.DOCKED
        if float(np.hypot(*self.state[:2])) > cfg.max_distance:
            return Outcome.ESCAPED
        if self.steps >= cfg.max_episode_steps:
            return Outcome.TIMEOUT
        return None
