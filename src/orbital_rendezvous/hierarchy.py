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
from typing import Any

import gymnasium as gym
import numpy as np

from .baselines import side_waypoint
from .env import EnvConfig, RendezvousEnv, goto_observation
from .rewards import Outcome

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
        planner: Callable[[np.ndarray], np.ndarray | list | None] | None = None,
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
        # The waypoint for a start state, a list of them, or None to fly
        # straight to the hold point: by default the V-bar procedure's rule.
        self.planner = planner or (lambda state: side_waypoint(
            state[:2], self.hold, self.keep_out, self.waypoint_distance))
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
        plan = self.planner(state)
        if plan is None:
            waypoints = []
        elif isinstance(plan, list):
            waypoints = [np.asarray(w, dtype=float) for w in plan]
        else:
            waypoints = [plan]
        self.plan = waypoints + [self.hold]
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


class PlannerEnv(gym.Env):
    """Choose the waypoint of an approach from a menu; frozen pilots fly the rest.

    One step per episode, a contextual bandit: the agent sees the start state
    and picks action 0, straight to the hold point, or ``k``, through waypoint
    ``k - 1`` of ``menu``; a `HierarchicalPilot` with that one-waypoint plan
    then flies the whole approach in the inner corridor environment.

    The reward is ``success_bonus`` on docking, ``failure_penalty`` otherwise,
    timeout included, less the cost of the approach,
    ``J = fuel_weight * delta_v + time_weight * T``. With ``relative``, the
    V-bar procedure's rule is flown from the same start too, and the reward
    counts ``J_rule - J`` instead of ``-J``: how much better than the rule the
    choice did from here. The cost of an approach depends mostly on where it
    starts, a tenth of a m/s between starts against a hundredth between
    choices; measured against the rule, the start cancels and what is left is
    the choice.

    The choice is discrete on purpose: behind the station the best side jumps
    at 180 degrees, and a categorical policy makes that jump where two of its
    smooth scores cross, while a continuous output would have to pass through
    straight ahead.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        corridor: RendezvousEnv,
        pilot: HierarchicalPilot,
        menu: np.ndarray,
        fuel_weight: float = 10.0,
        time_weight: float = 0.0,
        relative: bool = False,
    ) -> None:
        super().__init__()
        self.corridor, self.pilot, self.menu = corridor, pilot, np.asarray(menu, dtype=float)
        self.fuel_weight, self.time_weight, self.relative = fuel_weight, time_weight, relative
        # The pilot's own planner, the procedure's rule, before any choice replaces it.
        self.rule = pilot.planner
        self.config = corridor.config
        self.action_space = gym.spaces.Discrete(1 + len(self.menu))
        self.observation_space = gym.spaces.Box(
            np.array([-2.0, -2.0, -20.0, -20.0], dtype=np.float32),
            np.array([2.0, 2.0, 20.0, 20.0], dtype=np.float32),
        )
        self.steps = 0

    @property
    def state(self) -> np.ndarray:
        return self.corridor.state

    def waypoint(self, action: int) -> np.ndarray | None:
        return None if action == 0 else self.menu[action - 1]

    def cost(self, info: dict[str, Any]) -> float:
        """``J = fuel_weight * delta_v + time_weight * T`` of a flown approach."""
        return self.fuel_weight * info["delta_v"] + self.time_weight * info["time"]

    def _observation(self) -> np.ndarray:
        cfg = self.config
        scale = np.array([cfg.max_distance, cfg.max_distance,
                          cfg.velocity_scale, cfg.velocity_scale])
        obs = (self.corridor.state / scale).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.corridor.reset(seed=seed, options=options)
        self.steps = 0
        return self._observation(), {}

    def _fly(self, planner) -> dict[str, Any]:
        """The whole approach from the current start with ``planner``, with totals."""
        self.pilot.planner = planner
        obs, delta_v, violated, done = self.corridor._observation(), 0.0, False, False
        while not done:
            obs, _, terminated, truncated, info = self.corridor.step(self.pilot(self.corridor, obs))
            delta_v += info["delta_v"]
            violated = violated or info.get("keep_out_violated", False)
            done = terminated or truncated
        time = self.corridor.steps * self.config.time_step
        return dict(info, delta_v=delta_v, keep_out_violated=violated, time=time)

    def fly(self, action: int) -> dict[str, Any]:
        """The whole approach through the chosen waypoint."""
        chosen = self.waypoint(int(action))
        return dict(self._fly(lambda state: chosen), choice=int(action))

    def fly_rule(self, start: np.ndarray) -> dict[str, Any]:
        """The same approach from ``start`` with the procedure's rule."""
        self.corridor.state, self.corridor.steps = start.copy(), 0
        return self._fly(self.rule)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        start = self.corridor.state.copy()
        info = self.fly(action)
        docked = info["outcome"] is Outcome.DOCKED
        reward = (self.corridor.reward_config.success_bonus if docked
                  else self.corridor.reward_config.failure_penalty)
        reward -= self.cost(info)
        steps, final = self.corridor.steps, self.corridor.state.copy()
        if self.relative:
            rule = self.fly_rule(start)
            reward += self.cost(rule)
            info["rule_cost"] = self.cost(rule)
            # The episode's record is the chosen approach, not the rule's replay.
            self.corridor.state, self.corridor.steps = final, steps
        self.steps = steps
        info["is_success"] = docked
        return self._observation(), float(reward), True, False, info


def oracle_costs(env: PlannerEnv, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every choice of the menu flown from the start ``seed`` selects.

    Returns the planner's observation of that start, the cost ``J`` of each
    choice, and whether each one docked: the full information a bandit only
    sees one arm of, which the simulator can give for all of them.
    """
    costs, docked = np.zeros(env.action_space.n), np.zeros(env.action_space.n, dtype=bool)
    for action in range(env.action_space.n):
        obs, _ = env.reset(seed=seed)
        info = env.fly(action)
        costs[action], docked[action] = env.cost(info), info["outcome"] is Outcome.DOCKED
    return obs, costs, docked


def beam_search(
    env: PlannerEnv, seed: int, width: int = 3, depth: int = 3
) -> dict[tuple, dict[str, Any]]:
    """Plans of up to ``depth`` waypoints from the menu, flown from the start ``seed`` selects.

    Every one-waypoint plan and the plan with none are flown; then only the
    ``width`` cheapest plans that dock are extended by every other waypoint of
    the menu, and so on: ``1 + m + 2 w (m - 1)`` flights for ``m`` waypoints
    at depth 3, instead of ``m^3``. Returns each plan flown, a tuple of
    waypoints, with its cost ``J``, whether it docked, its time and delta-v.
    """
    menu = [tuple(float(c) for c in w) for w in env.menu]

    def fly(plan: tuple) -> dict[str, Any]:
        env.reset(seed=seed)
        info = env._fly(lambda state: [np.array(w) for w in plan])
        return {"cost": env.cost(info), "docked": info["outcome"] is Outcome.DOCKED,
                "time": info["time"], "delta_v": info["delta_v"]}

    flown = {(): fly(())}
    level = [(w,) for w in menu]
    for plan in level:
        flown[plan] = fly(plan)
    for _ in range(depth - 1):
        kept = sorted((p for p in level if flown[p]["docked"]),
                      key=lambda p: flown[p]["cost"])[:width]
        level = [p + (w,) for p in kept for w in menu if w != p[-1]]
        for plan in level:
            flown[plan] = fly(plan)
    return flown


def best_plan(flown: dict[tuple, dict[str, Any]], max_waypoints: int) -> tuple[tuple, dict]:
    """The cheapest plan that docks among those with at most ``max_waypoints`` waypoints."""
    candidates = [(p, v) for p, v in flown.items() if len(p) <= max_waypoints]
    return min(candidates, key=lambda pv: (not pv[1]["docked"], pv[1]["cost"]))
