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
- timeout: ``max_episode_steps`` reached. Since the agent sees the clock, the
  time limit is part of the task and the timeout is a true end of the episode,
  reported as ``terminated`` (Pardo et al., 2018). It is not penalised: the
  agent should not learn to fear the clock itself.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import gymnasium as gym
import numpy as np

from .dynamics import discretize, mean_motion, propagate
from .rewards import Outcome, RewardConfig, Scales, step_reward, terminal_reward


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
        )
        self.state = np.zeros(4)
        self.steps = 0

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
        angle = self.np_random.uniform(0.0, 2.0 * np.pi)
        velocity = self.np_random.normal(0.0, cfg.initial_velocity_scale, size=2)
        self.state = np.array(
            [radius * np.cos(angle), radius * np.sin(angle), velocity[0], velocity[1]]
        )
        self.steps = 0
        return self._observation(), self._info(None, np.zeros(2))

    def _outcome(self, previous_position: np.ndarray) -> Outcome | None:
        cfg = self.config
        distance = float(np.linalg.norm(self.state[:2]))
        speed = float(np.linalg.norm(self.state[2:]))

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

        outcome = self._outcome(previous_state[:2])
        # With the clock observed, every outcome, the timeout included, ends
        # the task; nothing is left to bootstrap from.
        terminated = outcome is not None
        truncated = False

        terms = step_reward(
            previous_state, self.state, thrust, terminated, self.reward_config, self.scales
        )
        terms["terminal"] = terminal_reward(outcome, self.reward_config)
        reward = float(sum(terms.values()))

        info = self._info(outcome, thrust)
        info["reward_terms"] = terms
        info["engine_on"] = engine_on
        return self._observation(), reward, terminated, truncated, info
