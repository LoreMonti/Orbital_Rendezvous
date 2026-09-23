"""The game view: one attempt drawn so that anyone can follow it.

The view from the target. The station sits at the centre, the Earth is below
and the orbit runs to the right. The chaser is an arrow pointing where it is
going, with its engine flame drawn out of the back, opposite to the thrust.
Rings mark distances in metres. A status bar above the scene shows the time,
the distance, the speed against the speed limit at that distance (the glide
slope of the reward), and the fuel used; it sits outside the scene so that it
never hides the chaser. Each attempt ends on a banner saying how it ended.

A `GameView` occupies one slot of a figure, so the training window can hold one
and the side-by-side comparison two, drawn identically by construction.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from .rewards import Outcome

# A dark "space" palette, shared by every figure of the project.
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


def heading_angle(screen_direction: np.ndarray) -> float:
    """Rotation in degrees that turns an upward triangle marker to ``screen_direction``."""
    sx, sy = screen_direction
    return float(np.degrees(np.arctan2(-sx, sy)))


def scene_extent(*position_sets: np.ndarray) -> float:
    """Half-width of a view that holds every trajectory given, with a margin."""
    reach = max(float(np.abs(p).max()) for p in position_sets if len(p))
    return max(40.0, 1.15 * reach)


def add_legend(ax, ncol: int = 3) -> None:
    """The legend of the game view, in its own axes so it never covers the scene."""
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
    ax.legend(handles=handles, loc="center", ncol=ncol, fontsize=9, facecolor=PANEL,
              edgecolor=GRID, labelcolor=TEXT, framealpha=1.0, columnspacing=1.6,
              handlelength=2.2, borderpad=0.9, title="What you are looking at",
              title_fontproperties={"size": 9, "weight": "bold"})
    ax.get_legend().get_title().set_color(MUTED)


class GameView:
    """The scene and its status bar, in the slot ``spec`` of ``fig``.

    Positions and velocities are given as ``(x, y)``, with ``x`` radial and
    ``y`` along-track; on screen the along-track axis is horizontal and the
    radial one vertical, with the Earth below.
    """

    def __init__(
        self,
        fig,
        spec,
        speed_limit: Callable[[float], float],
        docking_radius: float = 1.0,
        time_step: float = 10.0,
        mass: float = 500.0,
        max_thrust: float = 1.0,
        max_episode_steps: int = 300,
        trail_length: int = 300,
    ) -> None:
        self.speed_limit = speed_limit
        self.docking_radius = docking_radius
        self.time_step = time_step
        self.mass = mass
        self.max_thrust_norm = np.sqrt(2.0) * max_thrust
        # The most delta-v an attempt could spend, firing flat out on both axes
        # for its whole length: this is what the fuel gauge counts down from.
        self.fuel_capacity = self.max_thrust_norm * max_episode_steps * time_step / mass
        self.trail_length = trail_length
        self.hud: dict[str, float | bool] = {}

        slots = spec.subgridspec(2, 1, height_ratios=[1.0, 11.0], hspace=0.04)
        self.ax_hud = fig.add_subplot(slots[0])
        self.ax_plane = fig.add_subplot(slots[1])
        self._rings: list = []
        self._heading = 0.0
        self._setup_plane()
        self._setup_hud()

    @classmethod
    def from_env(cls, fig, spec, env, **kwargs) -> GameView:
        """Build the view from a `RendezvousEnv`, reusing its glide slope and scales."""
        from .rewards import speed_limit

        cfg = env.config
        return cls(
            fig,
            spec,
            speed_limit=lambda r: speed_limit(r, env.reward_config, env.scales),
            docking_radius=cfg.docking_radius,
            time_step=cfg.time_step,
            mass=cfg.mass,
            max_thrust=cfg.max_thrust,
            max_episode_steps=cfg.max_episode_steps,
            **kwargs,
        )

    # -- setup --------------------------------------------------------------------

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
        # Above the trail, on a dark tab, so the final approach cannot hide it.
        ax.annotate(
            "TARGET", (0.0, 0.0), xytext=(0, -22), textcoords="offset points", ha="center",
            color="#bfdbfe", fontsize=9, fontweight="bold", zorder=14,
            bbox={"boxstyle": "round,pad=0.2", "fc": SPACE, "ec": "none", "alpha": 0.8},
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

        self.banner = ax.text(
            0.5, 0.62, "", transform=ax.transAxes, ha="center", va="center", fontsize=34,
            fontweight="bold", zorder=10,
            bbox={"boxstyle": "round,pad=0.4", "fc": SPACE, "ec": "none", "alpha": 0.75},
        )
        self.banner.set_visible(False)

    def _setup_hud(self) -> None:
        """A status bar above the scene, like a video game's, so it never hides the chaser."""
        ax = self.ax_hud
        ax.set_facecolor(PANEL)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
        self.title = ax.set_title("waiting for the first attempt...", color=TEXT,
                                  fontsize=13, fontweight="bold", pad=8)

        # (label, left edge of its cell) across the bar.
        cells = (("TIME", 0.02), ("DISTANCE", 0.13), ("SPEED", 0.28),
                 ("SPEED LIMIT", 0.43), ("FUEL USED", 0.76))
        self._hud_values = {}
        for label, x in cells:
            ax.text(x, 0.73, label, transform=ax.transAxes, color=MUTED, fontsize=8,
                    fontweight="bold", va="center")
            self._hud_values[label] = ax.text(
                x, 0.3, "", transform=ax.transAxes, color=TEXT, fontsize=12,
                family="monospace", va="center",
            )
        bar = (0.76, 0.17, 0.08, 0.26)
        ax.add_patch(Rectangle(bar[:2], bar[2], bar[3], transform=ax.transAxes, fc=SPACE,
                               ec=GRID))
        self._fuel_bar = Rectangle(bar[:2], bar[2], bar[3], transform=ax.transAxes, fc=GREEN)
        self._fuel_width = bar[2]
        ax.add_patch(self._fuel_bar)
        self._hud_values["FUEL USED"].set_x(0.85)
        self._hud_values["FUEL USED"].set_fontsize(11)

    def _draw_rings(self, extent: float) -> None:
        for artist in self._rings:
            artist.remove()
        self._rings = []
        c = np.cos(np.pi / 4)
        for radius in RING_RADII:
            if radius < 0.12 * extent or radius > 1.4 * extent:
                continue
            ring = Circle((0.0, 0.0), radius, fill=False, ls=(0, (1, 3)), lw=0.9, color=GRID,
                          zorder=1)
            self.ax_plane.add_patch(ring)
            # Down the lower-right diagonal: each label sits at its own spot,
            # clear of the target. Rings reaching the edge go unlabelled rather
            # than cut in half.
            if c * radius > 0.8 * extent:
                self._rings.append(ring)
                continue
            label = self.ax_plane.text(c * radius, -c * radius, f" {radius:.0f} m", color=MUTED,
                                       fontsize=8, ha="left", va="top", zorder=1, clip_on=True)
            self._rings += [ring, label]

    # -- drawing ------------------------------------------------------------------

    def load(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        thrusts: np.ndarray,
        outcome: Outcome,
        title: str,
        extent: float | None = None,
    ) -> None:
        """Prepare an attempt for drawing; ``extent`` lets two views share one scale."""
        self.positions = np.asarray(positions, dtype=float)
        self.velocities = np.asarray(velocities, dtype=float)
        self.thrusts = np.asarray(thrusts, dtype=float)
        self.outcome = outcome
        self._title_text = title
        self.fuel_used = (
            np.cumsum(np.linalg.norm(self.thrusts, axis=1)) * self.time_step / self.mass
        )

        extent = extent or scene_extent(self.positions)
        self.ax_plane.set_xlim(-extent, extent)
        # Extra room at the bottom for the Earth.
        self.ax_plane.set_ylim(-1.2 * extent, extent)
        self._draw_rings(extent)
        self._flame_scale = 0.14 * extent / self.max_thrust_norm

        self._trail.set_color(BLUE)
        self._path.set_data(self.positions[:, 1], self.positions[:, 0])
        self.banner.set_visible(False)
        self.title.set_text(title)

    @property
    def n_steps(self) -> int:
        return len(self.positions)

    def draw(self, k: int) -> None:
        """Draw step ``k``; past the end of the attempt, hold its last step with the banner."""
        last = self.n_steps - 1
        self._draw_step(min(k, last))
        if k >= last:
            self.finish()

    def finish(self) -> None:
        color = OUTCOME_COLORS[self.outcome]
        self._trail.set_color(color)
        self.banner.set_text(BANNERS[self.outcome])
        self.banner.set_color(color)
        self.banner.set_visible(True)
        self.title.set_text(f"{self._title_text}  ·  {BANNERS[self.outcome].lower()}")

    def _draw_step(self, k: int) -> None:
        positions, velocities, thrusts = self.positions, self.velocities, self.thrusts
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
        used = float(self.fuel_used[k])
        fuel_left = max(0.0, 1.0 - used / self.fuel_capacity)
        time = (k + 1) * self.time_step
        self.hud = {
            "time": time, "distance": distance, "speed": speed, "limit": limit,
            "within_limit": within, "fuel_used": used, "fuel_left": fuel_left,
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
        self._hud_values["FUEL USED"].set_text(f"{used:.2f} m/s")
