"""Gymnasium environment for the planar rendezvous task.

Action: ``a`` in ``[-1, 1]^2``, mapped to the thrust ``u = u_max a`` and
saturated per axis. The thrust is continuous, so the agent can correct gently
instead of choosing between a few abrupt burns. Optionally, a thruster has a
minimum level: commands below ``thrust_deadzone`` (a fraction of ``u_max``) on
an axis leave that thruster off. Real thrusters cannot fire arbitrarily weakly,
and with it the exploration noise around zero no longer burns fuel, so the
agent can coast for free.

Optionally too, an engine switch: with ``engine_switch`` the action gains a
third component, ``a_on``, and the engine fires only when ``a_on > 0``. Off, the
thrust is exactly zero whatever the noise on the other two components, so
coasting costs nothing even while exploring; on, the thrust stays continuous
with no minimum level, keeping the fine control a minimum level would lose.

Observation: the relative state, normalised so that every component is of
order one, and the fraction of the episode elapsed,
``[x / r_max, y / r_max, vx / v_ref, vy / v_ref, t / T_max]``. Positions are
hundreds of metres and velocities centimetres per second; fed raw, the network
would barely see the velocities. Without the clock, two identical states early
and late in an episode would look the same to the agent although their futures
differ, and the problem would not be Markovian.

An episode ends in one of four ways:

- docked: inside the docking radius and slower than the docking speed;
- crashed: inside the docking radius, or through it, too fast;
- escaped: further than ``max_distance`` from the target;
- keep-out violation: with ``keep_out_radius`` set, entering the keep-out
  sphere around the station outside the approach cone, of half-angle
  ``approach_cone_deg`` around the docking axis +y (the V-bar). The docking
  sphere itself is exempt, as the cone's apex is the port. With
  ``keep_out_mode = "penalty"`` a violation does not end the episode: each step
  spent in the forbidden zone costs ``keep_out_weight`` instead, a constraint a
  Lagrange multiplier can price during training (see
  `callbacks.KeepOutBudget`), while evaluation keeps the strict rule;
- timeout: ``max_episode_steps`` reached. Since the agent sees the clock, the
  time limit is part of the task and the timeout is a true end of the episode,
  reported as ``terminated`` (Pardo et al., 2018). It is not penalised: the
  agent should not learn to fear the clock itself.

The chaser starts at a random distance in ``initial_radius_range``, in a random
direction. ``start_angle_range_deg`` can restrict that direction: the angle
between the start and the docking axis +y is drawn in the range, on either side
of the axis at random. The default, 0 to 180 degrees, is every direction, drawn
exactly as without the option; `callbacks.StartCurriculum` widens a narrower
range as the agent masters it, from starts in front of the port to starts
behind the station.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import gymnasium as gym
import numpy as np

from .dynamics import discretize, mean_motion, propagate
from .rewards import (
    Outcome,
    RewardConfig,
    Scales,
    in_approach_cone,
    step_reward,
    terminal_reward,
)


@dataclass(frozen=True)
class EnvConfig:
    """Physical and episode parameters. See ``configs/ppo_default.yaml``."""

    semi_major_axis: float = 6778.0e3
    mu: float = 3.986004418e14
    mass: float = 500.0
    max_thrust: float = 1.0
    time_step: float = 10.0
    max_episode_steps: int = 300
    initial_radius_range: tuple[float, float] = (80.0, 200.0)
    initial_velocity_scale: float = 0.05
    docking_radius: float = 1.0
    docking_speed: float = 0.05
    max_distance: float = 500.0
    velocity_scale: float = 0.5
    thrust_deadzone: float = 0.0
    engine_switch: bool = False
    keep_out_radius: float = 0.0
    approach_cone_deg: float = 15.0
    keep_out_mode: str = "terminal"
    keep_out_weight: float = 0.0
    start_angle_range_deg: tuple[float, float] = (0.0, 180.0)


def _closest_approach(p0: np.ndarray, p1: np.ndarray) -> float:
    """Distance from the origin to the segment ``p0 -> p1``.

    At a few metres per second the chaser can cross the docking sphere between
    two steps; checking only the endpoints would let it fly through the target.
    """
    d = p1 - p0
    length2 = float(d @ d)
    if length2 == 0.0:
        return float(np.linalg.norm(p0))
    t = np.clip(-(p0 @ d) / length2, 0.0, 1.0)
    return float(np.linalg.norm(p0 + t * d))


class RendezvousEnv(gym.Env):
    """A chaser spacecraft learning to dock with a target in the LVLH frame.

    The physical state ``[x, y, vx, vy]`` in SI units is kept in ``self.state``;
    the agent only ever sees its normalised version.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.reward_config = reward_config or RewardConfig()

        self.n = mean_motion(self.config.semi_major_axis, self.config.mu)
        self.phi, self.gamma = discretize(self.n, self.config.time_step, self.config.mass)

        # Thrust on the two axes, and with the engine switch a third command, a_on.
        n_actions = 3 if self.config.engine_switch else 2
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype=np.float32)
        # Finite but generous bounds: twice the escape radius on position, and
        # 10 m/s on velocity, more than the delta-v an episode can spend. The
        # elapsed fraction of the episode lies in [0, 1].
        low = np.array([-2.0, -2.0, -20.0, -20.0, 0.0], dtype=np.float32)
        high = np.array([2.0, 2.0, 20.0, 20.0, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        self._obs_scale = np.array(
            [
                self.config.max_distance,
                self.config.max_distance,
                self.config.velocity_scale,
                self.config.velocity_scale,
            ]
        )
        self.scales = Scales(
            max_distance=self.config.max_distance,
            velocity_scale=self.config.velocity_scale,
            docking_speed=self.config.docking_speed,
            mass=self.config.mass,
            time_step=self.config.time_step,
            keep_out_radius=self.config.keep_out_radius,
            approach_cone_deg=self.config.approach_cone_deg,
        )
        self.state = np.zeros(4)
        self.steps = 0
        # Where the shaping pulls to: the target at the origin here, a point
        # to reach in `GoToEnv`.
        self.goal = np.zeros(2)

    def set_approach_cone(self, degrees: float) -> None:
        """Change the half-angle of the approach cone, for a curriculum that narrows it."""
        self.config = replace(self.config, approach_cone_deg=degrees)
        self.scales = replace(self.scales, approach_cone_deg=degrees)

    def set_start_angles(self, low: float, high: float) -> None:
        """Change the range of start angles from the docking axis, for a curriculum on starts."""
        self.config = replace(self.config, start_angle_range_deg=(low, high))

    def set_keep_out_weight(self, weight: float) -> None:
        """Change the cost of a step in the forbidden zone, in penalty mode."""
        self.config = replace(self.config, keep_out_weight=weight)

    def set_fuel_weight(self, weight: float) -> None:
        """Change the cost of fuel, for a curriculum that raises it during training.

        Only the fuel term changes; the shaping potential does not, so the
        shaping stays a pure telescoping sum whatever the schedule.
        """
        self.reward_config = replace(self.reward_config, fuel_weight=weight)

    def _observation(self) -> np.ndarray:
        elapsed = self.steps / self.config.max_episode_steps
        obs = np.append(self.state / self._obs_scale, elapsed).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _info(self, outcome: Outcome | None, thrust: np.ndarray) -> dict[str, Any]:
        cfg = self.config
        info: dict[str, Any] = {
            "position": self.state[:2].copy(),
            "velocity": self.state[2:].copy(),
            "distance": float(np.linalg.norm(self.state[:2])),
            "speed": float(np.linalg.norm(self.state[2:])),
            "thrust": thrust,
            "delta_v": float(np.linalg.norm(thrust)) * cfg.time_step / cfg.mass,
            "outcome": outcome,
        }
        if outcome is not None:
            # Read by Stable-Baselines3 to log the success rate.
            info["is_success"] = outcome is Outcome.DOCKED
        return info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        cfg = self.config
        radius = self.np_random.uniform(*cfg.initial_radius_range)
        low, high = cfg.start_angle_range_deg
        if (low, high) == (0.0, 180.0):
            # Every direction, drawn as before the option existed, so that a
            # seed still selects the same start and old models reproduce.
            angle = self.np_random.uniform(0.0, 2.0 * np.pi)
            position = radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            # Angle from the docking axis +y, on a random side of it.
            angle = np.radians(self.np_random.uniform(low, high))
            side = 1.0 if self.np_random.random() < 0.5 else -1.0
            position = radius * np.array([side * np.sin(angle), np.cos(angle)])
        velocity = self.np_random.normal(0.0, cfg.initial_velocity_scale, size=2)
        self.state = np.append(position, velocity)
        self.steps = 0
        return self._observation(), self._info(None, np.zeros(2))

    def _violates_keep_out(self, p0: np.ndarray, p1: np.ndarray, samples: int = 21) -> bool:
        """Whether the step from ``p0`` to ``p1`` enters the keep-out sphere outside the cone.

        Checked along the whole segment, not only at its ends: in one step the
        chaser can cross the sphere with neither end inside it.
        """
        cfg = self.config
        for s in np.linspace(0.0, 1.0, samples):
            point = p0 + s * (p1 - p0)
            distance = float(np.hypot(*point))
            if cfg.docking_radius <= distance < cfg.keep_out_radius and not in_approach_cone(
                point, cfg.approach_cone_deg
            ):
                return True
        return False

    def _outcome(self, previous_position: np.ndarray, violated: bool) -> Outcome | None:
        cfg = self.config
        distance = float(np.linalg.norm(self.state[:2]))
        speed = float(np.linalg.norm(self.state[2:]))

        if violated and cfg.keep_out_mode == "terminal":
            return Outcome.KEEP_OUT
        # The segment test covers the endpoint too: ending inside the sphere is
        # the special case where the closest point is the last one.
        if _closest_approach(previous_position, self.state[:2]) < cfg.docking_radius:
            return Outcome.DOCKED if speed < cfg.docking_speed else Outcome.CRASHED
        if distance > cfg.max_distance:
            return Outcome.ESCAPED
        if self.steps >= cfg.max_episode_steps:
            return Outcome.TIMEOUT
        return None

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        command = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        engine_on = True
        if self.config.engine_switch:
            engine_on = bool(command[2] > 0.0)
            command = command[:2] if engine_on else np.zeros(2)
        # Below its minimum level a thruster stays off.
        command[np.abs(command) < self.config.thrust_deadzone] = 0.0
        thrust = self.config.max_thrust * command
        previous_state = self.state.copy()

        self.state = propagate(self.state, thrust, self.phi, self.gamma)
        self.steps += 1

        violated = bool(self.config.keep_out_radius) and self._violates_keep_out(
            previous_state[:2], self.state[:2]
        )
        outcome = self._outcome(previous_state[:2], violated)
        # With the clock observed, every outcome, the timeout included, ends
        # the task; nothing is left to bootstrap from.
        terminated = outcome is not None
        truncated = False

        # The shaping measures distances from the goal. At the origin, the
        # default, the shift is an exact zero and nothing changes.
        shift = np.append(self.goal, [0.0, 0.0])
        terms = step_reward(
            previous_state - shift, self.state - shift, thrust, terminated,
            self.reward_config, self.scales,
        )
        terms["terminal"] = terminal_reward(outcome, self.reward_config)
        if self.config.keep_out_radius:
            penalised = violated and self.config.keep_out_mode == "penalty"
            terms["keep_out"] = -self.config.keep_out_weight if penalised else 0.0
        reward = float(sum(terms.values()))

        info = self._info(outcome, thrust)
        info["reward_terms"] = terms
        info["engine_on"] = engine_on
        info["keep_out_violated"] = violated
        return self._observation(), reward, terminated, truncated, info


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
