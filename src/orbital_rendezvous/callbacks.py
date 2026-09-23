"""Stable-Baselines3 callback that feeds the live window during training."""

from __future__ import annotations

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

    The curves are redrawn at the end of every rollout, and the PPO losses are
    read at the start of the next one, after the update that produced them.
    """

    def __init__(
        self,
        live_view: LiveView,
        episode_stride: int = 50,
        env_index: int = 0,
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
        self._thrusts: list[np.ndarray] = []
        self._last_replay_at = 0

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

        positions, thrusts = np.array(self._positions), np.array(self._thrusts)
        self._positions, self._thrusts = [], []
        if self.curves.n_episodes - self._last_replay_at >= self.episode_stride:
            self._last_replay_at = self.curves.n_episodes
            self.replays += 1
            self.live_view.show_episode(
                positions, thrusts, outcome, self.curves.n_episodes, delta_v
            )

    def _on_rollout_start(self) -> None:
        values = self.model.logger.name_to_value
        if "train/policy_gradient_loss" in values:
            self.curves.policy_loss.append(float(values["train/policy_gradient_loss"]))
            self.curves.value_loss.append(float(values["train/value_loss"]))

    def _on_rollout_end(self) -> None:
        self.live_view.update_curves(self.curves)
