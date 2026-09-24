"""Stable-Baselines3 callback that feeds the live window during training."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .env import RendezvousEnv
from .live_view import LiveView, TrainingCurves
from .rewards import Outcome


class LiveViewCallback(BaseCallback):
    """Collect per-episode statistics and refresh the training window.

    At every step it accumulates, for each parallel environment, the true
    return (fuel plus terminal, without the shaping) and the delta-v spent, and
    records the trajectory of environment ``env_index``. When an episode ends
    its statistics join the curves; once every ``episode_stride`` episodes the
    last finished episode of ``env_index`` is replayed on the left panel. The
    episode shown is a real training episode, exploration noise included.

    The curves are redrawn at the end of every rollout.

    With ``record_at``, replays are also captured for the training GIF: the
    first replay at or after each of those episode counts, plus the last replay
    of the run, which shows what the agent finally learned.
    """

    def __init__(
        self,
        live_view: LiveView,
        episode_stride: int = 50,
        env_index: int = 0,
        record_at: Sequence[int] = (),
        capture_frames: int = 36,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.live_view = live_view
        self.episode_stride = episode_stride
        self.env_index = env_index
        self.curves = TrainingCurves()
        self.replays = 0
        self._returns: np.ndarray | None = None
        self._delta_v: np.ndarray | None = None
        self._positions: list[np.ndarray] = []
        self._velocities: list[np.ndarray] = []
        self._thrusts: list[np.ndarray] = []
        self._last_replay_at = 0
        self.record_at = sorted(record_at)
        self.capture_frames = capture_frames
        self.milestone_clips: list[list[np.ndarray]] = []
        self._last_clip: list[np.ndarray] = []
        self._last_clip_is_milestone = False
        self._next_milestone = 0

    def _reset_accumulators(self, n_envs: int) -> None:
        self._returns = np.zeros(n_envs)
        self._delta_v = np.zeros(n_envs)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        if self._returns is None or len(self._returns) != len(infos):
            self._reset_accumulators(len(infos))

        for i, info in enumerate(infos):
            terms = info["reward_terms"]
            self._returns[i] += terms["fuel"] + terms["terminal"]
            self._delta_v[i] += info["delta_v"]
            if i == self.env_index:
                self._positions.append(info["position"])
                self._velocities.append(info["velocity"])
                self._thrusts.append(info["thrust"])
            if dones[i]:
                self._finish_episode(i, info)
        return True

    def _finish_episode(self, i: int, info: dict) -> None:
        outcome = info["outcome"]
        delta_v = float(self._delta_v[i])
        self.curves.record_episode(outcome, float(self._returns[i]), delta_v)
        self._returns[i] = 0.0
        self._delta_v[i] = 0.0
        if i != self.env_index:
            return

        positions = np.array(self._positions)
        velocities = np.array(self._velocities)
        thrusts = np.array(self._thrusts)
        self._positions, self._velocities, self._thrusts = [], [], []
        if self.curves.n_episodes - self._last_replay_at >= self.episode_stride:
            self._last_replay_at = self.curves.n_episodes
            self.replays += 1
            # Every replay is captured when recording, since any could be the last.
            capture = self.capture_frames if self.record_at else 0
            clip = self.live_view.show_episode(
                positions, velocities, thrusts, outcome, self.curves.n_episodes, capture=capture
            )
            if capture:
                self._keep(clip)

    def _keep(self, clip: list[np.ndarray]) -> None:
        # Several thresholds crossed since the last replay still yield one clip.
        milestone = False
        while (self._next_milestone < len(self.record_at)
               and self.curves.n_episodes >= self.record_at[self._next_milestone]):
            self._next_milestone += 1
            milestone = True
        if milestone:
            self.milestone_clips.append(clip)
        self._last_clip = clip
        self._last_clip_is_milestone = milestone

    def recorded_clips(self) -> list[list[np.ndarray]]:
        """The milestone clips in order, then the last replay if it is not one of them."""
        clips = list(self.milestone_clips)
        if self._last_clip and not self._last_clip_is_milestone:
            clips.append(self._last_clip)
        return clips

    def _on_rollout_end(self) -> None:
        self.live_view.update_curves(self.curves)


class FuelCurriculum(BaseCallback):
    """Raise the fuel weight linearly from ``start`` to ``end`` over the first part of training.

    With a heavy fuel cost from the start, moving costs more than approaching
    earns before the docking bonus has ever been seen, and the agent learns to
    stay put. Starting light lets it learn to dock first; the weight then rises
    to the value that defines the task, and stays there for the rest of the run:

        w_f(t) = start + (end - start) * min(1, t / (ramp * total)).
    """

    def __init__(self, start: float, end: float, ramp: float = 0.5, verbose: int = 0) -> None:
        super().__init__(verbose)
        if not 0.0 < ramp <= 1.0:
            raise ValueError("ramp must be a fraction of the training, in (0, 1]")
        self.start, self.end, self.ramp = start, end, ramp
        self.weight = start

    def weight_at(self, progress: float) -> float:
        """Fuel weight once ``progress`` (0 to 1) of the training has elapsed."""
        return self.start + (self.end - self.start) * min(1.0, progress / self.ramp)

    def _update(self) -> None:
        total = self.model._total_timesteps
        self.weight = self.weight_at(self.num_timesteps / total if total else 1.0)
        self.training_env.env_method("set_fuel_weight", self.weight)
        self.logger.record("curriculum/fuel_weight", self.weight)

    def _on_training_start(self) -> None:
        self._update()

    def _on_rollout_start(self) -> None:
        self._update()

    def _on_step(self) -> bool:
        return True


class FuelBudget(BaseCallback):
    """A Lagrange multiplier on fuel: the fuel weight that keeps delta-v within ``budget``.

    The task becomes "dock, spending at most ``budget`` m/s", and the fuel weight
    is the multiplier of that constraint, adjusted by dual ascent: every few
    rollouts the agent flies ``episodes`` deterministic attempts, and

        lambda <- clip(lambda + step_size * (delta_v - budget) / budget, 0, max_weight).

    Spending too much raises the price of fuel, spending less lowers it. The
    measurement uses the deterministic policy, so exploration noise, which the
    trained agent does not pay, cannot push the price up.

    The multiplier stays at zero until the agent docks in at least
    ``warmup_success`` of those attempts: raising the price of fuel before the
    agent can dock would teach it to stay put, the trap of a heavy fuel cost.
    """

    def __init__(
        self,
        budget: float,
        make_env: Callable[[], RendezvousEnv],
        step_size: float = 1.0,
        evaluate_every: int = 10,
        episodes: int = 20,
        warmup_success: float = 0.5,
        max_weight: float = 50.0,
        first_seed: int = 50_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        if budget <= 0.0:
            raise ValueError("the fuel budget must be positive")
        self.budget = budget
        self.make_env = make_env
        self.step_size = step_size
        self.evaluate_every = evaluate_every
        self.episodes = episodes
        self.warmup_success = warmup_success
        self.max_weight = max_weight
        # Starts of their own, apart from training and from the held-out evaluation.
        self.seeds = range(first_seed, first_seed + episodes)
        self.weight = 0.0
        self.active = False
        self.history: list[dict[str, float]] = []
        self._env: RendezvousEnv | None = None
        self._rollouts = 0

    def dual_step(self, weight: float, delta_v: float) -> float:
        """One step of dual ascent on the multiplier."""
        weight += self.step_size * (delta_v - self.budget) / self.budget
        return float(np.clip(weight, 0.0, self.max_weight))

    def measure(self) -> tuple[float, float]:
        """Docking rate and mean delta-v of the deterministic policy on the budget's starts."""
        from .evaluation import evaluate

        if self._env is None:
            self._env = self.make_env()
        runs = evaluate(
            self._env, lambda e, obs: self.model.predict(obs, deterministic=True)[0], self.seeds
        )
        success = float(np.mean([r.outcome is Outcome.DOCKED for r in runs]))
        return success, float(np.mean([r.delta_v for r in runs]))

    def _apply(self) -> None:
        self.training_env.env_method("set_fuel_weight", self.weight)
        self.logger.record("budget/fuel_weight", self.weight)

    def _on_training_start(self) -> None:
        self._apply()

    def _on_rollout_end(self) -> None:
        self._rollouts += 1
        if self._rollouts % self.evaluate_every:
            return
        success, delta_v = self.measure()
        if not self.active and success >= self.warmup_success:
            self.active = True
        if self.active:
            self.weight = self.dual_step(self.weight, delta_v)
            self._apply()
        self.logger.record("budget/delta_v", delta_v)
        self.logger.record("budget/success", success)
        self.history.append({
            "timesteps": int(self.num_timesteps), "success": success, "delta_v": delta_v,
            "fuel_weight": self.weight, "active": self.active,
        })

    def _on_step(self) -> bool:
        return True
