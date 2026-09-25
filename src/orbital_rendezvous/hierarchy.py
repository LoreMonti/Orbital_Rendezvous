"""Two learned pilots flying the plan of the V-bar procedure.

On an oriented target, a single agent learned to hold the approach corridor
but not to go around the station from every direction, and it lost manoeuvres
it had learned (Step 15). Here the plan stays the classical one, and only the
flying is learned:

1. the planner of the V-bar procedure, `baselines.side_waypoint`, puts a
   waypoint beside the keep-out sphere if the straight way to the hold point
   crosses it, on the side the chaser starts from;
2. a go-to agent (`env.GoToEnv`) flies to the waypoint, passing it without
   stopping, then to the hold point 30 m out on the docking axis, where it
   stops;
3. once within the hold radius and slower than the hold speed, a final-approach
   agent, trained on starts around the hold point with the full rule, flies
   down the corridor to the port.

Where one would need a network to jump, choosing left or right behind the
station, the planner decides; each network only has to learn a smooth map,
on a task that never changed during its training. Each agent sees the clock of
its own leg, as in its training.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .baselines import side_waypoint
from .env import EnvConfig, RendezvousEnv, goto_observation

Policy = Callable[[np.ndarray], np.ndarray]
FINAL_LOW = RendezvousEnv().observation_space.low
FINAL_HIGH = RendezvousEnv().observation_space.high


class HierarchicalPilot:
    """Planner plus two learned pilots, callable as a controller, ``(env, obs) -> action``.

    ``go_to`` and ``final`` map an observation to an action. ``go_to_config``
    and ``final_config`` are the environment configurations the two agents
    were trained with: their observations are normalised, and their clocks
    run, as in training.
    """

    def __init__(
        self,
        go_to: Policy,
        final: Policy,
        go_to_config: EnvConfig,
        final_config: EnvConfig,
        keep_out: float = 20.0,
        hold_distance: float = 30.0,
        pass_radius: float = 5.0,
        waypoint_distance: float = 50.0,
    ) -> None:
        self.go_to, self.final = go_to, final
        self.go_to_config, self.final_config = go_to_config, final_config
        self.keep_out = keep_out
        self.hold = np.array([0.0, hold_distance])
        self.pass_radius = pass_radius
        # The handover is where the go-to agent was trained to stop.
        self.hold_radius = go_to_config.docking_radius
        self.hold_speed = go_to_config.docking_speed
        self.waypoint_distance = waypoint_distance
        self.reset(np.zeros(4))

    @classmethod
    def from_models(cls, go_to_model, final_model, go_to_config: EnvConfig,
                    final_config: EnvConfig, **kwargs) -> HierarchicalPilot:
        """From two trained Stable-Baselines3 models, flown deterministically."""
        return cls(
            lambda obs: go_to_model.predict(obs, deterministic=True)[0],
            lambda obs: final_model.predict(obs, deterministic=True)[0],
            go_to_config, final_config, **kwargs,
        )

    def reset(self, state: np.ndarray) -> None:
        waypoint = side_waypoint(state[:2], self.hold, self.keep_out, self.waypoint_distance)
        self.plan = ([waypoint] if waypoint is not None else []) + [self.hold]
        self.leg = 0
        self.phase = "fly"
        self.leg_start = 0

    def _elapsed(self, steps: int, config: EnvConfig) -> float:
        return min(1.0, (steps - self.leg_start) / config.max_episode_steps)

    def __call__(self, env: RendezvousEnv, obs: np.ndarray) -> np.ndarray:
        state = env.state
        if env.steps == 0:
            self.reset(state)
        position, speed = state[:2], float(np.hypot(*state[2:]))
        if self.phase == "fly":
            goal = self.plan[self.leg]
            distance = float(np.hypot(*(position - goal)))
            if self.leg < len(self.plan) - 1 and distance < self.pass_radius:
                self.leg += 1
                self.leg_start = env.steps
            elif (self.leg == len(self.plan) - 1 and distance < self.hold_radius
                  and speed < self.hold_speed):
                self.phase = "final"
                self.leg_start = env.steps
        if self.phase == "fly":
            goal = self.plan[self.leg]
            elapsed = self._elapsed(env.steps, self.go_to_config)
            return self.go_to(goto_observation(state, goal, elapsed, self.go_to_config))
        cfg = self.final_config
        scale = np.array([cfg.max_distance, cfg.max_distance,
                          cfg.velocity_scale, cfg.velocity_scale])
        observation = np.append(state / scale, self._elapsed(env.steps, cfg)).astype(np.float32)
        # Clipped to the bounds of the environment it was trained in.
        return self.final(np.clip(observation, FINAL_LOW, FINAL_HIGH))
