"""A sampling planner on the known dynamics: model predictive control by search.

At every decision the planner imagines many thrust sequences over a horizon of
``H`` steps, flies them in the exact Clohessy-Wiltshire model, scores each and
applies the first thrust of the best; ten seconds later it plans again. The
search is the cross-entropy method: sequences drawn around a mean, the best
``elites`` of them refit the mean and the spread, a few times over.

The score of a sequence is the return the environment would pay along it, with
the same outcomes in the same order (keep-out violation, docking or crash,
escape), its fuel cost, and, if the horizon ends before the episode does, a
value of the state it reaches:

    J = sum_k gamma^k r_k + gamma^H V(s_H).

``V`` is where knowledge of what lies beyond the horizon enters. Step 26 of the
ROADMAP compares three that are not learned: none, ``V = 0``; the straight
distance to the port, ``V = -w r / r_max``, what a generic planner would use,
blind to the corridor; and the shaping potential, which measures the way
around the keep-out sphere, knowledge written by hand. The fourth, learned
from the planner's own flights, is passed as ``value_fn`` (see `value`).

The thrust is held over blocks of ``block`` steps while sampling, so a
sequence over 150 steps is 30 numbers, not 300; the mean keeps one thrust per
step, shifted by one step after every decision so that each plan starts from
the last.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from orbital_rendezvous.core.env import RendezvousEnv
from orbital_rendezvous.core.rewards import potential

VALUES = ("none", "distance", "potential", "learned", "graph")


@dataclass(frozen=True)
class PlannerConfig:
    """Search settings. See the module docstring."""

    horizon: int = 50          # steps simulated ahead
    block: int = 5             # steps over which a sampled thrust is held
    samples: int = 500         # sequences per iteration
    elites: int = 50           # best sequences that refit the distribution
    iterations: int = 5        # refits per decision
    init_std: float = 0.5      # spread of the first iteration, in units of max thrust
    value: str = "distance"    # beyond the horizon: none, distance, potential or learned
    value_scale: float = 1.0   # multiplies V, against the +-100 of the outcomes
    segment_samples: int = 5   # points per step checked against the keep-out rule


class SamplingPlanner:
    """Cross-entropy planner on the environment's own model; a `Controller`.

    Reads the true state, as the classical controllers do, and returns an
    action in ``[-1, 1]^2``. Call `reset` at the start of every episode.
    """

    def __init__(
        self,
        env: RendezvousEnv,
        config: PlannerConfig | None = None,
        seed: int = 0,
        value_fn: Callable[[np.ndarray, int], np.ndarray] | None = None,
    ):
        self.config = config or PlannerConfig()
        if self.config.value not in VALUES:
            raise ValueError(f"unknown value {self.config.value!r}; known: {VALUES}")
        if (self.config.value in ("learned", "graph")) != (value_fn is not None):
            raise ValueError("a learned or graph value needs value_fn, and only they do")
        self.value_fn = value_fn
        if env.config.mirror_symmetry or env.config.engine_switch:
            raise ValueError("the planner commands the true thrust on two axes")
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.mean = np.zeros((self.config.horizon, 2))

    def reset(self) -> None:
        self.mean = np.zeros((self.config.horizon, 2))

    def __call__(self, env: RendezvousEnv, obs: np.ndarray) -> np.ndarray:
        if env.steps == 0:
            self.reset()
        cfg = self.config
        blocks = -(-cfg.horizon // cfg.block)
        std = np.full((blocks, 2), cfg.init_std)
        for _ in range(cfg.iterations):
            noise = self.rng.normal(size=(cfg.samples, blocks, 2)) * std
            noise = np.repeat(noise, cfg.block, axis=1)[:, : cfg.horizon]
            actions = np.clip(self.mean + noise, -1.0, 1.0)
            actions[0] = np.clip(self.mean, -1.0, 1.0)      # the last plan competes too
            scores = self.score(env.state, actions, env.steps)
            elite = actions[np.argsort(scores)[-cfg.elites:]]
            self.mean = elite.mean(axis=0)
            # The spread of the elites over each block, kept from collapsing.
            pad = blocks * cfg.block - cfg.horizon
            padded = np.concatenate([elite, np.repeat(elite[:, -1:], pad, axis=1)], axis=1)
            std = np.maximum(padded.reshape(cfg.elites, blocks, cfg.block, 2).std(axis=(0, 2)),
                             0.05)
        action = self.mean[0].copy()
        self.mean = np.vstack([self.mean[1:], self.mean[-1:]])
        return action

    def score(self, state: np.ndarray, actions: np.ndarray, steps_done: int) -> np.ndarray:
        """Discounted return of each sequence of ``actions`` (N, H, 2) from ``state``."""
        env, cfg = self.env, self.config
        ec, rc = env.config, env.reward_config
        n = len(actions)
        states = np.repeat(state[None, :], n, axis=0)
        total = np.zeros(n)
        alive = np.ones(n, dtype=bool)
        discount = 1.0
        fractions = np.linspace(0.0, 1.0, cfg.segment_samples)
        steps_left = ec.max_episode_steps - steps_done
        horizon = min(cfg.horizon, steps_left)
        for k in range(horizon):
            thrust = ec.max_thrust * actions[:, k]
            new = states @ env.phi.T + thrust @ env.gamma.T
            delta_v = np.linalg.norm(thrust, axis=1) * ec.time_step / ec.mass
            reward = -rc.fuel_weight * delta_v
            p0, p1 = states[:, :2], new[:, :2]
            violated = np.zeros(n, dtype=bool)
            if ec.keep_out_radius:
                for s in fractions:
                    point = p0 + s * (p1 - p0)
                    r = np.hypot(point[:, 0], point[:, 1])
                    inside = (r >= ec.docking_radius) & (r < ec.keep_out_radius)
                    cone = point[:, 1] >= r * np.cos(np.radians(ec.approach_cone_deg))
                    violated |= inside & ~cone
            d = p1 - p0
            t = np.clip(-(p0 * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0.0, 1.0)
            closest = np.hypot(*(p0 + t[:, None] * d).T)
            speed = np.hypot(new[:, 2], new[:, 3])
            arrived = (closest < ec.docking_radius) & ~violated
            docked = arrived & (speed < ec.docking_speed)
            failed = violated | (arrived & ~docked) | (np.hypot(*p1.T) > ec.max_distance)
            reward = reward + np.where(docked, rc.success_bonus, 0.0)
            reward = reward + np.where(failed, rc.failure_penalty, 0.0)
            total += alive * discount * reward
            alive &= ~(docked | failed)
            states = new
            discount *= rc.gamma
        if horizon < steps_left:
            total += alive * discount * cfg.value_scale * self.value(states, steps_done + horizon)
        return total

    def value(self, states: np.ndarray, steps: int) -> np.ndarray:
        """``V`` beyond the horizon, for a batch of states reached after ``steps`` steps."""
        kind = self.config.value
        if kind in ("learned", "graph"):
            return self.value_fn(states, steps)
        if kind == "none":
            return np.zeros(len(states))
        rc, scales = self.env.reward_config, self.env.scales
        if kind == "distance":
            return -rc.distance_weight * np.hypot(states[:, 0], states[:, 1]) / scales.max_distance
        return np.array([potential(s, rc, scales) for s in states])
