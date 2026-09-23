"""The window watched during training, designed to read like a video game.

Left panel: the view from the target. The station sits at the centre, the Earth
is below and the orbit runs to the right. The chaser is an arrow pointing where
it is going, with its engine flame drawn out of the back, opposite to the
thrust. Rings mark distances in metres. A HUD shows the time, the
distance, the speed against the speed limit at that distance (the glide slope
of the reward), and a fuel gauge, in a status bar above the scene so that it
never hides the chaser. Each replay ends on a banner saying how the
attempt ended. An attempt is replayed sped up, about 130 frames whatever its
length.

Right panel: the quantities that actually mean something in reinforcement
learning, with plain-language titles, and below them the legend of the game
view, kept off the scene so it never hides the chaser. The docking rate and the
true score (fuel plus terminal, without the shaping term) are the curves that
show learning, followed by the fuel spent. The shaped return is not plotted:
with a negative potential and ``gamma < 1`` a step spent standing still earns
``(gamma - 1) Phi > 0``, so it rewards wandering. The PPO losses are not plotted
either: in reinforcement learning they do not say whether the agent is
improving, and Stable-Baselines3 logs them anyway.

Drawing every attempt would slow training down by orders of magnitude, so the
left panel is replayed once every ``episode_stride`` attempts, while the curves
are refreshed after every rollout, which costs almost nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import matplotlib
import matplotlib.ticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from .rewards import Outcome

NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "ps", "svg", "cairo", "template"}

# A dark "space" palette, shared by both panels.
BACKGROUND = "#0b1020"
SPACE = "#04060d"
PANEL = "#111827"
TEXT = "#e5e7eb"
MUTED = "#9ca3af"
GRID = "#334155"
GREEN = "#34d399"
RED = "#f87171"
AMBER = "#fbbf24"
BLUE = "#60a5fa"
ORANGE = "#fb923c"
CHASER = "#f9fafb"

OUTCOME_COLORS = {
    Outcome.DOCKED: GREEN,
    Outcome.CRASHED: RED,
    Outcome.ESCAPED: RED,
    Outcome.TIMEOUT: MUTED,
}
BANNERS = {
    Outcome.DOCKED: "DOCKED!",
    Outcome.CRASHED: "CRASHED",
    Outcome.ESCAPED: "LOST IN SPACE",
    Outcome.TIMEOUT: "OUT OF TIME",
}
RING_RADII = (10.0, 25.0, 50.0, 100.0, 200.0, 400.0)


@dataclass
class TrainingCurves:
    """The histories shown on the right panel, one entry per episode."""

    outcomes: list[Outcome] = field(default_factory=list)
    true_return: list[float] = field(default_factory=list)
    delta_v: list[float] = field(default_factory=list)

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


def heading_angle(screen_direction: np.ndarray) -> float:
    """Rotation in degrees that turns an upward triangle marker to ``screen_direction``."""
    sx, sy = screen_direction
    return float(np.degrees(np.arctan2(-sx, sy)))


class LiveView:
    """A single matplotlib figure with the game view and the training curves.

    Positions and velocities are given as ``(x, y)``, with ``x`` radial and
    ``y`` along-track; on screen the along-track axis is horizontal and the
    radial one vertical, with the Earth below.
    """

    def __init__(
        self,
        max_distance: float,
        docking_radius: float,
        speed_limit: Callable[[float], float],
        time_step: float = 1.0,
        mass: float = 500.0,
        max_thrust: float = 1.0,
        max_episode_steps: int = 2000,
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
        self.time_step = time_step
        self.mass = mass
        self.max_thrust_norm = np.sqrt(2.0) * max_thrust
        # The most delta-v an attempt could spend, firing flat out on both axes
        # for its whole length: this is what the fuel gauge counts down from.
        self.fuel_capacity = self.max_thrust_norm * max_episode_steps * time_step / mass
        self.trail_length = trail_length
        self.success_window = success_window
        self.replay_frames = replay_frames
        self.frame_interval = frame_interval
        self.interactive = matplotlib.get_backend().lower() not in NON_INTERACTIVE_BACKENDS
        self.hud: dict[str, float | bool] = {}

        if self.interactive:
            plt.ion()
        self.fig = plt.figure(figsize=(14.0, 7.6), facecolor=BACKGROUND)
        self.fig.canvas.manager.set_window_title("Orbital Rendezvous: training")
        grid = self.fig.add_gridspec(
            4, 2, width_ratios=[1.2, 1.0], height_ratios=[3.0, 3.0, 2.4, 1.0],
            wspace=0.18, hspace=0.62, left=0.03, right=0.94, top=0.93, bottom=0.07,
        )
        left = grid[:, 0].subgridspec(2, 1, height_ratios=[1.0, 11.0], hspace=0.04)
        self.ax_hud = self.fig.add_subplot(left[0])
        self.ax_plane = self.fig.add_subplot(left[1])
        self.ax_success = self.fig.add_subplot(grid[0, 1])
        self.ax_return = self.fig.add_subplot(grid[1, 1], sharex=self.ax_success)
        self.ax_dv = self.fig.add_subplot(grid[2, 1], sharex=self.ax_success)
        self.ax_legend = self.fig.add_subplot(grid[3, 1])

        self._rings: list = []
        self._heading = 0.0
        self._setup_plane()
        self._setup_hud()
        self._setup_curves()
        self._setup_legend()
        if self.interactive:
            plt.show(block=False)

    @classmethod
    def from_env(cls, env, **kwargs) -> LiveView:
        """Build the view from a `RendezvousEnv`, reusing its glide slope and scales."""
        from .rewards import speed_limit

        cfg = env.config
        return cls(
            max_distance=cfg.max_distance,
            docking_radius=cfg.docking_radius,
            speed_limit=lambda r: speed_limit(r, env.reward_config, env.scales),
            time_step=cfg.time_step,
            mass=cfg.mass,
            max_thrust=cfg.max_thrust,
            max_episode_steps=cfg.max_episode_steps,
            **kwargs,
        )

    # -- the game view ------------------------------------------------------------

    def _setup_plane(self) -> None:
        ax = self.ax_plane
        ax.set_facecolor(SPACE)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)

        # Stars and the Earth are fixed to the window, not to the metres, so
        # they stay put when the view zooms from one attempt to the next.
        rng = np.random.default_rng(7)
        ax.scatter(
            rng.random(160), rng.random(160), s=rng.random(160) * 2.2, color="white",
            alpha=0.55, transform=ax.transAxes, zorder=0, linewidths=0,
        )
        ax.add_patch(Circle((0.5, -2.62), 2.7, transform=ax.transAxes, color="#1d4ed8",
                            alpha=0.18, zorder=0))
        ax.add_patch(Circle((0.5, -2.62), 2.68, transform=ax.transAxes, color="#1e3a8a",
                            zorder=0))
        ax.text(0.5, 0.025, "▼  EARTH, 400 km below", transform=ax.transAxes, ha="center",
                color="#bfdbfe", fontsize=10, fontweight="bold", zorder=1)
        ax.annotate(
            "", xy=(0.97, 0.965), xytext=(0.72, 0.965), xycoords="axes fraction",
            arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.5},
        )
        ax.text(0.845, 0.935, "direction of orbit", transform=ax.transAxes, ha="center",
                color=MUTED, fontsize=9)

        ax.add_patch(Circle((0.0, 0.0), self.docking_radius, color=GREEN, alpha=0.35, zorder=3))
        ax.plot(0.0, 0.0, marker="_", ms=30, mew=7, color="#93c5fd", zorder=4)
        ax.plot(0.0, 0.0, marker="s", ms=10, color="#e5e7eb", mec="#475569", zorder=5)
        self._target_label = ax.annotate(
            "TARGET", (0.0, 0.0), xytext=(0, -20), textcoords="offset points", ha="center",
            color="#bfdbfe", fontsize=9, fontweight="bold", zorder=5,
        )

        (self._path,) = ax.plot([], [], lw=1.0, ls=(0, (2, 2)), color=MUTED, alpha=0.6, zorder=2)
        # The chaser, its flame and its recent trail are drawn above the banner,
        # so they stay visible wherever it flies.
        (self._trail,) = ax.plot([], [], lw=2.4, color=BLUE, zorder=11)
        (self._flame_outer,) = ax.plot([], [], lw=6.0, color=ORANGE, solid_capstyle="round",
                                       alpha=0.9, zorder=12)
        (self._flame_inner,) = ax.plot([], [], lw=2.5, color=AMBER, solid_capstyle="round",
                                       zorder=12)
        (self._chaser,) = ax.plot([], [], ls="none", marker=(3, 0, 0), ms=15, color=CHASER,
                                  mec="black", zorder=13)

        self._banner = ax.text(
            0.5, 0.62, "", transform=ax.transAxes, ha="center", va="center", fontsize=34,
            fontweight="bold", zorder=10,
            bbox={"boxstyle": "round,pad=0.4", "fc": SPACE, "ec": "none", "alpha": 0.75},
        )
        self._banner.set_visible(False)

    def _setup_legend(self) -> None:
        """The legend of the game view, laid out wide under the curves."""
        ax = self.ax_legend
        ax.axis("off")
        handles = [
            Line2D([], [], ls="none", marker=(3, 0, 0), ms=11, color=CHASER, mec="black",
                   label="chaser, pointing where it goes"),
            Line2D([], [], ls="none", marker="s", ms=9, color="#e5e7eb", mec="#475569",
                   label="target station"),
            Line2D([], [], color=ORANGE, lw=5, label="engine firing"),
            Line2D([], [], ls="none", marker="o", ms=11, color=GREEN, alpha=0.5,
                   label="docking zone"),
            Line2D([], [], color=BLUE, lw=2.4, label="recent path"),
            Line2D([], [], color=MUTED, lw=1.2, ls=(0, (2, 2)), label="whole path"),
        ]
        ax.legend(handles=handles, loc="center", ncol=3, fontsize=9, facecolor=PANEL,
                  edgecolor=GRID, labelcolor=TEXT, framealpha=1.0, columnspacing=1.6,
                  handlelength=2.2, borderpad=0.9, title="What you are looking at",
                  title_fontproperties={"size": 9, "weight": "bold"})
        ax.get_legend().get_title().set_color(MUTED)

    def _setup_hud(self) -> None:
        """A status bar above the scene, like a video game's, so it never hides the chaser."""
        ax = self.ax_hud
        ax.set_facecolor(PANEL)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
        self._title = ax.set_title("waiting for the first attempt...", color=TEXT,
                                   fontsize=13, fontweight="bold", pad=8)

        # (label, left edge of its cell) across the bar.
        cells = (("TIME", 0.02), ("DISTANCE", 0.15), ("SPEED", 0.33),
                 ("SPEED LIMIT", 0.51), ("FUEL", 0.78))
        self._hud_values = {}
        for label, x in cells:
            ax.text(x, 0.73, label, transform=ax.transAxes, color=MUTED, fontsize=8,
                    fontweight="bold", va="center")
            self._hud_values[label] = ax.text(
                x, 0.3, "", transform=ax.transAxes, color=TEXT, fontsize=12,
                family="monospace", va="center",
            )
        bar = (0.78, 0.17, 0.12, 0.26)
        ax.add_patch(Rectangle(bar[:2], bar[2], bar[3], transform=ax.transAxes, fc=SPACE,
                               ec=GRID))
        self._fuel_bar = Rectangle(bar[:2], bar[2], bar[3], transform=ax.transAxes, fc=GREEN)
        self._fuel_width = bar[2]
        ax.add_patch(self._fuel_bar)
        self._hud_values["FUEL"].set_x(0.91)
        self._hud_values["FUEL"].set_fontsize(11)

    def _draw_rings(self, extent: float) -> None:
        for artist in self._rings:
            artist.remove()
        self._rings = []
        for radius in RING_RADII:
            if radius < 0.12 * extent or radius > 1.4 * extent:
                continue
            ring = Circle((0.0, 0.0), radius, fill=False, ls=(0, (1, 3)), lw=0.9, color=GRID,
                          zorder=1)
            self.ax_plane.add_patch(ring)
            # Down the lower-right diagonal: each label sits at its own spot,
            # clear of the target.
            c = np.cos(np.pi / 4)
            label = self.ax_plane.text(c * radius, -c * radius, f" {radius:.0f} m", color=MUTED,
                                       fontsize=8, ha="left", va="top", zorder=1, clip_on=True)
            self._rings += [ring, label]

    def show_episode(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        thrusts: np.ndarray,
        outcome: Outcome,
        episode: int,
    ) -> None:
        """Replay an attempt, sped up to about ``replay_frames`` frames, ending on a banner."""
        positions = np.asarray(positions, dtype=float)
        velocities = np.asarray(velocities, dtype=float)
        thrusts = np.asarray(thrusts, dtype=float)
        if len(positions) == 0:
            return

        extent = max(40.0, 1.15 * float(np.abs(positions).max()))
        self.ax_plane.set_xlim(-extent, extent)
        # Extra room at the bottom for the Earth.
        self.ax_plane.set_ylim(-1.2 * extent, extent)
        self._draw_rings(extent)
        self._flame_scale = 0.14 * extent / self.max_thrust_norm

        color = OUTCOME_COLORS[outcome]
        self._trail.set_color(BLUE)
        self._path.set_data(positions[:, 1], positions[:, 0])
        self._banner.set_visible(False)
        self._title.set_text(f"ATTEMPT {episode}")
        fuel_used = np.cumsum(np.linalg.norm(thrusts, axis=1)) * self.time_step / self.mass

        frames = np.unique(np.linspace(0, len(positions) - 1, self.replay_frames).astype(int))
        if not self.interactive:
            frames = frames[-1:]
        for k in frames:
            self._draw_frame(positions, velocities, thrusts, fuel_used, k)
            self.refresh(self.frame_interval)

        self._trail.set_color(color)
        self._banner.set_text(BANNERS[outcome])
        self._banner.set_color(color)
        self._banner.set_visible(True)
        self._title.set_text(f"ATTEMPT {episode}  ·  {BANNERS[outcome].lower()}")
        self.refresh(0.8 if self.interactive else 0.0)

    def _draw_frame(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        thrusts: np.ndarray,
        fuel_used: np.ndarray,
        k: int,
    ) -> None:
        start = max(0, k + 1 - self.trail_length)
        self._trail.set_data(positions[start : k + 1, 1], positions[start : k + 1, 0])

        x, y = positions[k]
        vx, vy = velocities[k]
        if np.hypot(vx, vy) > 1e-6:
            self._heading = heading_angle(np.array([vy, vx]))
        self._chaser.set_data([y], [x])
        self._chaser.set_marker((3, 0, self._heading))

        # The flame leaves the engine opposite to the thrust.
        ux, uy = thrusts[k]
        fy, fx = -self._flame_scale * uy, -self._flame_scale * ux
        self._flame_outer.set_data([y, y + fy], [x, x + fx])
        self._flame_inner.set_data([y, y + 0.6 * fy], [x, x + 0.6 * fx])

        distance = float(np.hypot(x, y))
        speed = float(np.hypot(vx, vy))
        limit = float(self.speed_limit(distance))
        within = speed <= limit
        fuel_left = max(0.0, 1.0 - float(fuel_used[k]) / self.fuel_capacity)
        time = (k + 1) * self.time_step
        self.hud = {
            "time": time, "distance": distance, "speed": speed, "limit": limit,
            "within_limit": within, "fuel_left": fuel_left,
        }

        minutes, seconds = divmod(int(round(time)), 60)
        self._hud_values["TIME"].set_text(f"{minutes:02d}:{seconds:02d}")
        self._hud_values["DISTANCE"].set_text(f"{distance:.1f} m")
        self._hud_values["SPEED"].set_text(f"{speed:.2f} m/s")
        self._hud_values["SPEED"].set_color(GREEN if within else RED)
        self._hud_values["SPEED LIMIT"].set_text(
            f"{limit:.2f} m/s " + ("✓ OK" if within else "✗ TOO FAST")
        )
        self._hud_values["SPEED LIMIT"].set_color(GREEN if within else RED)
        self._fuel_bar.set_width(self._fuel_width * fuel_left)
        self._fuel_bar.set_color(GREEN if fuel_left > 0.5 else AMBER if fuel_left > 0.2 else RED)
        self._hud_values["FUEL"].set_text(f"{100 * fuel_left:3.0f}%")

    # -- the curves ---------------------------------------------------------------

    def _style(self, ax, title: str | None = None, ylabel: str | None = None) -> None:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(alpha=0.25, color=GRID)
        if title:
            ax.set_title(title, color=TEXT, fontsize=10, loc="left")
        if ylabel:
            ax.set_ylabel(ylabel, color=MUTED, fontsize=9)

    def _setup_curves(self) -> None:
        self._style(self.ax_success,
                    f"How often it docks  (share of the last {self.success_window} attempts)")
        self.ax_success.set_ylim(-0.03, 1.03)
        self.ax_success.yaxis.set_major_formatter(
            matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
        )
        self._style(self.ax_return, "Score  (docking bonus minus fuel spent, no hints)")
        self._style(self.ax_dv, "Fuel spent per attempt", ylabel=r"$\Delta v$ [m/s]")
        self.ax_dv.set_xlabel("attempt", color=MUTED, fontsize=9)
        self.plt.setp(self.ax_success.get_xticklabels(), visible=False)
        self.plt.setp(self.ax_return.get_xticklabels(), visible=False)

        (self._success_line,) = self.ax_success.plot([], [], lw=2.4, color=GREEN)
        self._return_dots = self.ax_return.scatter([], [], s=5, color="#475569", linewidths=0)
        (self._return_line,) = self.ax_return.plot([], [], lw=2.4, color=BLUE)
        (self._dv_line,) = self.ax_dv.plot([], [], lw=1.8, color=ORANGE)

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
        self.refresh()

    # -- plumbing -----------------------------------------------------------------

    def refresh(self, pause: float = 0.001) -> None:
        """Push the changes to the screen; a no-op beyond drawing when headless."""
        if self.interactive:
            self.fig.canvas.draw_idle()
            self.plt.pause(max(pause, 0.001))
        else:
            self.fig.canvas.draw()

    def save(self, path: str) -> None:
        self.fig.savefig(path, dpi=120, facecolor=BACKGROUND)

    def close(self) -> None:
        self.plt.close(self.fig)
