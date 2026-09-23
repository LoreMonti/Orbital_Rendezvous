"""The window watched during training: the chaser on the left, progress on the right.

Left panel: the LVLH plane, with the target at the origin, the chaser, its
trail and its thrust, and reference circles labelled with the glide-slope speed
limit at that distance. An episode is replayed sped up, about 130 frames
whatever its length, and its trail is coloured by how it ended.

Right panel: the quantities that actually mean something in reinforcement
learning. The success rate and the *true* episode return (fuel plus terminal,
without the shaping term) are the curves that show learning. The shaped return
is not plotted: with a negative potential and ``gamma < 1`` a step spent
standing still earns ``(gamma - 1) Phi > 0``, so it rewards wandering. The PPO
losses are drawn small and secondary, because a policy loss hovering around
zero, or a value loss that grows once the agent starts docking, is normal.

Drawing every episode would slow training down by orders of magnitude, so the
left panel is replayed once every ``episode_stride`` episodes, while the curves
are refreshed after every rollout, which costs almost nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import matplotlib
import numpy as np

from .rewards import Outcome

NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "ps", "svg", "cairo", "template"}

OUTCOME_COLORS = {
    Outcome.DOCKED: "#2a9d5c",
    Outcome.CRASHED: "#d1495b",
    Outcome.ESCAPED: "#d1495b",
    Outcome.TIMEOUT: "#8d8d8d",
}


@dataclass
class TrainingCurves:
    """The histories shown on the right panel: one entry per episode or per update."""

    outcomes: list[Outcome] = field(default_factory=list)
    true_return: list[float] = field(default_factory=list)
    delta_v: list[float] = field(default_factory=list)
    policy_loss: list[float] = field(default_factory=list)
    value_loss: list[float] = field(default_factory=list)

    @property
    def n_episodes(self) -> int:
        return len(self.outcomes)

    def record_episode(self, outcome: Outcome, true_return: float, delta_v: float) -> None:
        self.outcomes.append(outcome)
        self.true_return.append(true_return)
        self.delta_v.append(delta_v)

    def success(self) -> np.ndarray:
        return np.array([o is Outcome.DOCKED for o in self.outcomes], dtype=float)


def rolling_mean(values: np.ndarray | list[float], window: int) -> np.ndarray:
    """Trailing mean over ``window`` points, over fewer at the start of the series."""
    x = np.asarray(values, dtype=float)
    cumulative = np.concatenate([[0.0], np.cumsum(x)])
    end = np.arange(1, len(x) + 1)
    start = np.maximum(0, end - window)
    return (cumulative[end] - cumulative[start]) / (end - start)


class LiveView:
    """A single matplotlib figure with the trajectory and the training curves."""

    def __init__(
        self,
        max_distance: float,
        docking_radius: float,
        speed_limit: Callable[[float], float],
        trail_length: int = 300,
        success_window: int = 50,
        replay_frames: int = 130,
        frame_interval: float = 0.02,
    ) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.max_distance = max_distance
        self.docking_radius = docking_radius
        self.speed_limit = speed_limit
        self.trail_length = trail_length
        self.success_window = success_window
        self.replay_frames = replay_frames
        self.frame_interval = frame_interval
        self.interactive = matplotlib.get_backend().lower() not in NON_INTERACTIVE_BACKENDS

        if self.interactive:
            plt.ion()
        self.fig = plt.figure(figsize=(13.0, 7.0))
        grid = self.fig.add_gridspec(
            4, 2, width_ratios=[1.15, 1.0], height_ratios=[3.0, 3.0, 2.0, 1.6],
            wspace=0.25, hspace=0.45,
        )
        self.ax_plane = self.fig.add_subplot(grid[:, 0])
        self.ax_success = self.fig.add_subplot(grid[0, 1])
        self.ax_return = self.fig.add_subplot(grid[1, 1], sharex=self.ax_success)
        self.ax_dv = self.fig.add_subplot(grid[2, 1], sharex=self.ax_success)
        self.ax_loss = self.fig.add_subplot(grid[3, 1])
        self.ax_value = self.ax_loss.twinx()

        self._setup_plane()
        self._setup_curves()
        if self.interactive:
            plt.show(block=False)

    @classmethod
    def from_env(cls, env, **kwargs) -> LiveView:
        """Build the view from a `RendezvousEnv`, reusing its glide slope."""
        from .rewards import speed_limit

        return cls(
            max_distance=env.config.max_distance,
            docking_radius=env.config.docking_radius,
            speed_limit=lambda r: speed_limit(r, env.reward_config, env.scales),
            **kwargs,
        )

    # -- left panel ---------------------------------------------------------------

    def _setup_plane(self) -> None:
        ax = self.ax_plane
        ax.set_aspect("equal")
        ax.set_xlabel(r"along-track $y$ [m]")
        ax.set_ylabel(r"radial $x$ [m]")
        ax.grid(alpha=0.2)

        for radius in (25.0, 50.0, 100.0, 200.0, 400.0):
            circle = self.plt.Circle((0.0, 0.0), radius, fill=False, ls=":", lw=0.8, color="0.6")
            ax.add_patch(circle)
            ax.annotate(
                f"{self.speed_limit(radius):.2f} m/s",
                (radius * np.cos(np.pi / 4), radius * np.sin(np.pi / 4)),
                fontsize=7, color="0.5", clip_on=True,
            )
        ax.add_patch(
            self.plt.Circle((0.0, 0.0), self.docking_radius, color="#2a9d5c", alpha=0.4)
        )
        ax.plot(0.0, 0.0, marker="s", color="black", ms=6, zorder=5)

        (self._path,) = ax.plot([], [], lw=0.8, color="0.75")
        (self._trail,) = ax.plot([], [], lw=2.0, color="#3a6ea5")
        (self._chaser,) = ax.plot([], [], marker="o", color="#3a6ea5", ms=7, zorder=6)
        (self._thrust,) = ax.plot([], [], lw=2.0, color="#e07a1f", zorder=6)
        self._title = ax.set_title("waiting for the first episode", fontsize=11)

    def show_episode(
        self,
        positions: np.ndarray,
        thrusts: np.ndarray,
        outcome: Outcome,
        episode: int,
        delta_v: float,
    ) -> None:
        """Replay an episode on the left panel, sped up to about ``replay_frames`` frames.

        Positions are ``(x, y)`` with ``x`` radial; they are drawn with the
        along-track axis horizontal, the usual picture of a rendezvous.
        """
        positions = np.asarray(positions, dtype=float)
        thrusts = np.asarray(thrusts, dtype=float)
        if len(positions) == 0:
            return

        extent = max(50.0, 1.1 * float(np.abs(positions).max()))
        self.ax_plane.set_xlim(-extent, extent)
        self.ax_plane.set_ylim(-extent, extent)
        arrow = 0.12 * extent / max(float(np.abs(thrusts).max()), 1e-12)

        color = OUTCOME_COLORS[outcome]
        self._trail.set_color(color)
        self._chaser.set_color(color)
        self._path.set_data(positions[:, 1], positions[:, 0])

        frames = np.unique(np.linspace(0, len(positions) - 1, self.replay_frames).astype(int))
        if not self.interactive:
            frames = frames[-1:]
        for k in frames:
            self._draw_frame(positions, thrusts, k, arrow)
            self._title.set_text(
                f"episode {episode}  ·  step {k + 1}/{len(positions)}  ·  "
                f"{outcome.value}  ·  $\\Delta v$ = {delta_v:.2f} m/s"
            )
            self.refresh(self.frame_interval)

    def _draw_frame(self, positions: np.ndarray, thrusts: np.ndarray, k: int, arrow: float) -> None:
        start = max(0, k + 1 - self.trail_length)
        self._trail.set_data(positions[start : k + 1, 1], positions[start : k + 1, 0])
        x, y = positions[k]
        self._chaser.set_data([y], [x])
        ux, uy = thrusts[k]
        self._thrust.set_data([y, y + arrow * uy], [x, x + arrow * ux])

    # -- right panel --------------------------------------------------------------

    def _setup_curves(self) -> None:
        self.ax_success.set_ylabel("success rate")
        self.ax_success.set_ylim(-0.03, 1.03)
        self.ax_success.set_title(
            f"learning progress (mean of the last {self.success_window} episodes)", fontsize=11
        )
        self.ax_return.set_ylabel("true return\n(fuel + terminal)")
        self.ax_dv.set_ylabel(r"$\Delta v$ [m/s]")
        self.ax_dv.set_xlabel("episode")
        self.ax_loss.set_xlabel("PPO update")
        self.ax_loss.set_ylabel("policy loss", fontsize=8, color="#6d597a")
        self.ax_value.set_ylabel("value loss", fontsize=8, color="#b56576")
        for ax in (self.ax_success, self.ax_return, self.ax_dv, self.ax_loss):
            ax.grid(alpha=0.2)
        for ax in (self.ax_loss, self.ax_value):
            ax.tick_params(labelsize=7)
        self.plt.setp(self.ax_success.get_xticklabels(), visible=False)
        self.plt.setp(self.ax_return.get_xticklabels(), visible=False)

        (self._success_line,) = self.ax_success.plot([], [], lw=2.2, color="#2a9d5c")
        self._return_dots = self.ax_return.scatter([], [], s=4, color="0.7")
        (self._return_line,) = self.ax_return.plot([], [], lw=2.2, color="#3a6ea5")
        (self._dv_line,) = self.ax_dv.plot([], [], lw=1.6, color="#e07a1f")
        (self._policy_line,) = self.ax_loss.plot([], [], lw=1.0, color="#6d597a")
        (self._value_line,) = self.ax_value.plot([], [], lw=1.0, color="#b56576")

    def update_curves(self, curves: TrainingCurves) -> None:
        """Redraw the right panel from the full histories."""
        w = self.success_window
        if curves.n_episodes:
            episodes = np.arange(1, curves.n_episodes + 1)
            self._success_line.set_data(episodes, rolling_mean(curves.success(), w))
            self._return_dots.set_offsets(np.column_stack([episodes, curves.true_return]))
            self._return_line.set_data(episodes, rolling_mean(curves.true_return, w))
            self._dv_line.set_data(episodes, rolling_mean(curves.delta_v, w))
            self.ax_success.set_xlim(0, max(10, curves.n_episodes))
            for ax, values in ((self.ax_return, curves.true_return), (self.ax_dv, curves.delta_v)):
                low, high = float(np.min(values)), float(np.max(values))
                pad = 0.05 * max(high - low, 1.0)
                ax.set_ylim(low - pad, high + pad)
        if curves.policy_loss:
            updates = np.arange(1, len(curves.policy_loss) + 1)
            self._policy_line.set_data(updates, curves.policy_loss)
            self._value_line.set_data(updates, curves.value_loss)
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()
            self.ax_value.relim()
            self.ax_value.autoscale_view()
        self.refresh()

    # -- plumbing -----------------------------------------------------------------

    def refresh(self, pause: float = 0.001) -> None:
        """Push the changes to the screen; a no-op beyond drawing when headless."""
        if self.interactive:
            self.fig.canvas.draw_idle()
            self.plt.pause(pause)
        else:
            self.fig.canvas.draw()

    def save(self, path: str) -> None:
        self.fig.savefig(path, dpi=120, bbox_inches="tight")

    def close(self) -> None:
        self.plt.close(self.fig)
