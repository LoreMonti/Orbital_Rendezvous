"""Stable-Baselines3 callbacks that feed the live window during training."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from .live_view import LiveView, TrainingCurves


class LiveViewCallback(BaseCallback):
    """Collect rollout statistics and refresh the training window.

    The trajectory panel is redrawn once every ``episode_stride`` episodes; the
    curves are refreshed after every policy update.
    """

    def __init__(
        self,
        live_view: LiveView,
        episode_stride: int = 50,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.live_view = live_view
        self.episode_stride = episode_stride
        self.curves = TrainingCurves()

    def _on_step(self) -> bool:
        raise NotImplementedError

    def _on_rollout_end(self) -> None:
        raise NotImplementedError
