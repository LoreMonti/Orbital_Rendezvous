"""The window watched during training: the chaser on the left, progress on the right.

Left panel: the LVLH plane, with the target at the origin, the chaser and its
trail, and the current thrust vector.

Right panel: the quantities that actually mean something in reinforcement
learning. The mean episode reward and the success rate are the curves that show
learning; the PPO losses are drawn small and secondary, because a policy loss
hovering around zero (or a value loss that grows once the agent starts reaching
the docking bonus) is normal and is not a sign of failure.

Drawing every rollout would slow training down by orders of magnitude, so the
left panel is refreshed once every ``episode_stride`` episodes while the curves
on the right are updated on every policy update, which costs almost nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrainingCurves:
    """The scalar histories shown on the right panel."""

    episode_rewards: list[float] = field(default_factory=list)
    success_rate: list[float] = field(default_factory=list)
    delta_v: list[float] = field(default_factory=list)
    policy_loss: list[float] = field(default_factory=list)
    value_loss: list[float] = field(default_factory=list)


class LiveView:
    """A single matplotlib figure with the trajectory and the training curves."""

    def __init__(self, max_distance: float, trail_length: int = 300) -> None:
        raise NotImplementedError

    def update_trajectory(
        self,
        positions: np.ndarray,
        thrust: np.ndarray,
        episode: int,
        delta_v: float,
    ) -> None:
        """Redraw the left panel with the latest rollout."""
        raise NotImplementedError

    def update_curves(self, curves: TrainingCurves) -> None:
        """Redraw the right panel."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
