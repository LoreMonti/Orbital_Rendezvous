"""Stable-Baselines3 callback that feeds the live window during training."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .live_view import LiveView, TrainingCurves


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
