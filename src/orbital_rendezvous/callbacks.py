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


class KeepOutBudget(BaseCallback):
    """A Lagrange multiplier on the keep-out constraint: the price of a forbidden step.

    The environment runs in penalty mode, where a step spent in the keep-out
    sphere outside the approach cone costs ``lambda`` instead of ending the
    episode. Every few rollouts the agent flies ``episodes`` deterministic
    attempts, and with ``p`` the fraction of them that entered the forbidden
    zone,

        lambda <- clip(lambda + step_size * (p - tolerance), 0, max_weight).

    A terminal penalty from the start teaches the agent to keep away from the
    station altogether: nearly every early approach comes in from the wrong
    side. So the price stays at zero until the agent docks in at least
    ``warmup_success`` of its attempts, as for the fuel budget. And if docking
    later falls below that level, raising the price further would only teach
    the agent to stay away, so the price relaxes instead, by ``relax`` per
    measurement: the task is to dock without violating, not to avoid
    violations at any cost.
    """

    def __init__(
        self,
        make_env: Callable[[], RendezvousEnv],
        tolerance: float = 0.02,
        step_size: float = 5.0,
        relax: float = 0.9,
        evaluate_every: int = 10,
        episodes: int = 20,
        warmup_success: float = 0.5,
        max_weight: float = 200.0,
        first_seed: int = 60_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.make_env = make_env
        self.tolerance = tolerance
        self.step_size = step_size
        self.relax = relax
        self.evaluate_every = evaluate_every
        self.episodes = episodes
        self.warmup_success = warmup_success
        self.max_weight = max_weight
        self.seeds = range(first_seed, first_seed + episodes)
        self.weight = 0.0
        self.active = False
        self.history: list[dict[str, float]] = []
        self._env: RendezvousEnv | None = None
        self._rollouts = 0

    def dual_step(self, weight: float, violation_rate: float) -> float:
        """One step of dual ascent on the multiplier."""
        weight += self.step_size * (violation_rate - self.tolerance)
        return float(np.clip(weight, 0.0, self.max_weight))

    def measure(self) -> tuple[float, float]:
        """Docking rate and violation rate of the deterministic policy."""
        from .evaluation import evaluate

        if self._env is None:
            self._env = self.make_env()
        runs = evaluate(
            self._env, lambda e, obs: self.model.predict(obs, deterministic=True)[0], self.seeds
        )
        success = float(np.mean([r.outcome is Outcome.DOCKED for r in runs]))
        return success, float(np.mean([r.violated for r in runs]))

    def _apply(self) -> None:
        self.training_env.env_method("set_keep_out_weight", self.weight)
        self.logger.record("keep_out/weight", self.weight)

    def _on_training_start(self) -> None:
        self._apply()

    def _on_rollout_end(self) -> None:
        self._rollouts += 1
        if self._rollouts % self.evaluate_every:
            return
        success, violation_rate = self.measure()
        if not self.active and success >= self.warmup_success:
            self.active = True
        if self.active:
            if success >= self.warmup_success:
                self.weight = self.dual_step(self.weight, violation_rate)
            else:
                self.weight *= self.relax
            self._apply()
        self.logger.record("keep_out/violation_rate", violation_rate)
        self.logger.record("keep_out/success", success)
        self.history.append({
            "timesteps": int(self.num_timesteps), "success": success,
            "violation_rate": violation_rate, "weight": self.weight, "active": self.active,
        })

    def _on_step(self) -> bool:
        return True


class MasteryCurriculum(BaseCallback):
    """A task made harder one stage at a time, whenever the agent masters the current one.

    Every ``evaluate_every`` rollouts the deterministic policy flies
    ``episodes`` attempts on an evaluation environment set to the current stage,
    with the strict rule, where a violation ends the attempt. If it docks in at
    least ``success_threshold`` of them, the stage advances; otherwise it stays.
    Subclasses say what a stage is: ``advanced`` moves ``value`` on towards
    ``final_deg``, ``_configure`` sets the evaluation environment to it and
    ``_apply`` the training ones. Once the final stage is reached the
    measurements stop. With ``after``, another curriculum, this one waits for
    that one to finish before measuring anything: two curricula in sequence.
    """

    #: Name of the stage in ``history``, and prefix of the logged keys.
    key = "stage"
    prefix = "curriculum"

    def __init__(
        self,
        make_env: Callable[[], RendezvousEnv],
        value: float,
        success_threshold: float = 0.9,
        evaluate_every: int = 10,
        episodes: int = 20,
        first_seed: int = 70_000,
        after: MasteryCurriculum | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.make_env = make_env
        self.value = value
        self.final_deg = value
        self.after = after
        self.success_threshold = success_threshold
        self.evaluate_every = evaluate_every
        self.episodes = episodes
        self.seeds = range(first_seed, first_seed + episodes)
        self.history: list[dict[str, float]] = []
        self._env: RendezvousEnv | None = None
        self._rollouts = 0

    @property
    def finished(self) -> bool:
        return self.value == self.final_deg

    @property
    def active(self) -> bool:
        """Whether this curriculum is measuring: not finished, and not waiting for another."""
        return not self.finished and (self.after is None or self.after.finished)

    def advanced(self, value: float, success: float) -> float:
        raise NotImplementedError

    def _configure(self, env: RendezvousEnv) -> None:
        raise NotImplementedError

    def _apply(self) -> None:
        raise NotImplementedError

    def measure(self) -> float:
        """Docking rate of the deterministic policy at the current stage, strict rule."""
        from .evaluation import evaluate

        if self._env is None:
            self._env = self.make_env()
        self._configure(self._env)
        runs = evaluate(
            self._env, lambda e, obs: self.model.predict(obs, deterministic=True)[0], self.seeds
        )
        return float(np.mean([r.outcome is Outcome.DOCKED for r in runs]))

    def _on_training_start(self) -> None:
        self._apply()

    def _on_rollout_end(self) -> None:
        if not self.active:
            return
        self._rollouts += 1
        if self._rollouts % self.evaluate_every:
            return
        success = self.measure()
        self.value = self.advanced(self.value, success)
        self._apply()
        self.logger.record(f"{self.prefix}/success", success)
        self.history.append({
            "timesteps": int(self.num_timesteps), "success": success, self.key: self.value,
        })

    def _on_step(self) -> bool:
        return True


class ConeCurriculum(MasteryCurriculum):
    """Narrow the approach cone from ``start_deg`` to ``final_deg`` as the agent masters it.

    Each stage narrows the cone by ``step_deg``. At 180 degrees the cone is the
    whole plane and there is no constraint at all, so training starts as for
    the default agent.

    Imposing the final cone at once, as a terminal rule or as a price on
    violations, made the agent stop approaching altogether: coming in through
    the port is a different strategy from coming in from wherever it starts.
    Narrowing the cone a little at a time asks for a small correction of the
    strategy it already has, and only once it has mastered the current one.
    In Step 14 this reached 90 degrees and no further.
    """

    key = "cone_deg"
    prefix = "cone"

    def __init__(
        self,
        make_env: Callable[[], RendezvousEnv],
        final_deg: float = 15.0,
        start_deg: float = 180.0,
        step_deg: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(make_env, start_deg, **kwargs)
        self.final_deg = final_deg
        self.step_deg = step_deg

    @property
    def cone_deg(self) -> float:
        return self.value

    def narrowed(self, cone_deg: float, success: float) -> float:
        """The cone after one measurement: narrower if the agent mastered this one."""
        if success >= self.success_threshold:
            return max(self.final_deg, cone_deg - self.step_deg)
        return cone_deg

    advanced = narrowed

    def _configure(self, env: RendezvousEnv) -> None:
        env.set_approach_cone(self.value)

    def _apply(self) -> None:
        self.training_env.env_method("set_approach_cone", self.value)
        self.logger.record("cone/half_angle_deg", self.value)


class StartCurriculum(MasteryCurriculum):
    """Widen the starts from in front of the port to behind the station, as the agent masters them.

    What changes is where the chaser starts: at an angle from the docking axis
    between 0 and ``value``, first ``start_deg``, widened by ``step_deg`` at each
    stage up to ``final_deg``, while the approach cone stays at its final angle.
    Each stage asks for a way around the keep-out sphere only a little longer
    than the last.

    On its own, from scratch, it never docks, not even from in front of the
    port: the Coriolis term pushes an approach along the V-bar sideways, by
    ``2 n |y'|``, out of a 15 degree cone, and an agent that cannot dock yet does
    not find the correction by chance. It therefore runs ``after`` a
    `ConeCurriculum` that narrows the cone on the first stage's starts: first
    docking, then the corridor, then the way around the station.

    Training starts are drawn over the whole current range, so the easy starts
    are not forgotten, but mastery is measured only on the outer ``band_deg``:
    at 180 degrees the newest 10 degrees are 6 % of the range, and a threshold
    of 90 % over all of it could be passed while failing every new start.
    """

    key = "start_deg"
    prefix = "starts"

    def __init__(
        self,
        make_env: Callable[[], RendezvousEnv],
        start_deg: float = 15.0,
        final_deg: float = 180.0,
        step_deg: float = 10.0,
        band_deg: float = 30.0,
        first_seed: int = 80_000,
        **kwargs,
    ) -> None:
        super().__init__(make_env, start_deg, first_seed=first_seed, **kwargs)
        self.final_deg = final_deg
        self.step_deg = step_deg
        self.band_deg = band_deg

    def widened(self, start_deg: float, success: float) -> float:
        """The range of starts after one measurement: wider if the agent mastered this one."""
        if success >= self.success_threshold:
            return min(self.final_deg, start_deg + self.step_deg)
        return start_deg

    advanced = widened

    def _configure(self, env: RendezvousEnv) -> None:
        env.set_start_angles(max(0.0, self.value - self.band_deg), self.value)

    def _apply(self) -> None:
        self.training_env.env_method("set_start_angles", 0.0, self.value)
        self.logger.record("starts/max_angle_deg", self.value)
