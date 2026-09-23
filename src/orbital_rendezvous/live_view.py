"""The window watched during training: the game view on the left, progress on the right.

Left: a `GameView` (see `game_view`) replaying one real training attempt, sped
up to about 130 frames whatever its length.

Right: the quantities that actually mean something in reinforcement learning,
with plain-language titles, and below them the legend of the game view, kept
off the scene so it never hides the chaser. The docking rate and the true score
(fuel plus terminal, without the shaping term) are the curves that show
learning, followed by the fuel spent. The shaped return is not plotted: with a
negative potential and ``gamma < 1`` a step spent standing still earns
``(gamma - 1) Phi > 0``, so it rewards wandering. The PPO losses are not
plotted either: in reinforcement learning they do not say whether the agent is
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

from .game_view import (
    AMBER,
    BACKGROUND,
    BLUE,
    CHASER,
    GREEN,
    GRID,
    MUTED,
    ORANGE,
    PANEL,
    RED,
    SPACE,
    TEXT,
    GameView,
    add_legend,
)
from .rewards import Outcome

# Every colour the views use, so that the training GIF keeps each one exactly.
THEME_COLORS = (
    BACKGROUND, SPACE, PANEL, TEXT, MUTED, GRID, GREEN, RED, AMBER, BLUE, ORANGE, CHASER,
    "#1e3a8a", "#93c5fd", "#bfdbfe", "#475569",
)

NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "ps", "svg", "cairo", "template"}


def is_interactive() -> bool:
    """Whether the current matplotlib backend draws to a window."""
    return matplotlib.get_backend().lower() not in NON_INTERACTIVE_BACKENDS


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


class LiveView:
    """A single matplotlib figure with the game view and the training curves."""

    def __init__(
        self,
        max_distance: float,
        docking_radius: float,
        speed_limit: Callable[[float], float],
        time_step: float = 10.0,
        mass: float = 500.0,
        max_thrust: float = 1.0,
        max_episode_steps: int = 300,
        trail_length: int = 300,
        success_window: int = 50,
        replay_frames: int = 130,
        frame_interval: float = 0.02,
    ) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.max_distance = max_distance
        self.success_window = success_window
        self.replay_frames = replay_frames
        self.frame_interval = frame_interval
        self.interactive = is_interactive()

        if self.interactive:
            plt.ion()
        self.fig = plt.figure(figsize=(14.0, 7.6), facecolor=BACKGROUND)
        self.fig.canvas.manager.set_window_title("Orbital Rendezvous: training")
        grid = self.fig.add_gridspec(
            4, 2, width_ratios=[1.2, 1.0], height_ratios=[3.0, 3.0, 2.4, 1.0],
            wspace=0.18, hspace=0.62, left=0.03, right=0.94, top=0.93, bottom=0.07,
        )
        self.game = GameView(
            self.fig, grid[:, 0], speed_limit, docking_radius, time_step, mass, max_thrust,
            max_episode_steps, trail_length,
        )
        self.ax_success = self.fig.add_subplot(grid[0, 1])
        self.ax_return = self.fig.add_subplot(grid[1, 1], sharex=self.ax_success)
        self.ax_dv = self.fig.add_subplot(grid[2, 1], sharex=self.ax_success)
        self.ax_legend = self.fig.add_subplot(grid[3, 1])

        self._setup_curves()
        add_legend(self.ax_legend)
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

    @property
    def hud(self) -> dict[str, float | bool]:
        """The values on the status bar at the last frame drawn."""
        return self.game.hud

    def show_episode(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        thrusts: np.ndarray,
        outcome: Outcome,
        episode: int,
        capture: int = 0,
    ) -> list[np.ndarray]:
        """Replay an attempt, sped up to about ``replay_frames`` frames, ending on a banner.

        With ``capture`` > 0, also returns about that many images of the whole
        window, evenly spread over the replay and ending on the banner: the
        raw material of the training GIF. Capturing works headless too.
        """
        if len(positions) == 0:
            return []
        self.game.load(positions, velocities, thrusts, outcome, f"ATTEMPT {episode}")
        last = self.game.n_steps - 1
        if self.interactive:
            drawn = self.replay_frames
        elif capture:
            drawn = capture
        else:
            drawn = 1
        # A single frame is the last step, the one with the banner.
        frames = np.unique(np.linspace(0, last, drawn).astype(int)) if drawn > 1 else [last]
        keep = set()
        if capture:
            keep = set(np.unique(np.linspace(0, len(frames) - 1, capture).astype(int)))

        images = []
        for i, k in enumerate(frames):
            self.game.draw(k)
            self.refresh(self.frame_interval)
            if i in keep:
                images.append(self.snapshot())
        self.refresh(0.8 if self.interactive else 0.0)
        if capture:
            # The banner is drawn by the refresh above, after the last step.
            images[-1] = self.snapshot()
        return images

    def snapshot(self, width: int = 960) -> np.ndarray:
        """The window as an RGB image ``width`` pixels wide.

        Shrunk straight away: on a high-density screen the canvas has twice
        the pixels, and a clip of full-size frames would take gigabytes.
        """
        from PIL import Image

        self.fig.canvas.draw()
        image = Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())[..., :3])
        height = round(image.height * width / image.width)
        return np.asarray(image.resize((width, height), Image.LANCZOS))

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


def save_gif(clips: list[list[np.ndarray]], path: str, fps: int = 12, hold: int = 18) -> None:
    """Join replay clips into one looping GIF, holding each clip's last frame for a moment."""
    from PIL import Image

    frames = []
    for clip in clips:
        frames += clip + [clip[-1]] * hold
    images = [Image.fromarray(frame) for frame in frames]
    # One shared palette keeps the colours steady from frame to frame. An
    # automatic palette is dominated by the dark background and loses the
    # accents, so the exact colours of the theme are added to it by hand: the
    # banners, the flame and the curves come out in their true colours.
    samples = [Image.fromarray(clip[i]) for clip in clips for i in (len(clip) // 2, -1)]
    width, height = samples[0].size
    mosaic = Image.new("RGB", (width * len(samples), height))
    for i, sample in enumerate(samples):
        mosaic.paste(sample, (i * width, 0))
    automatic = mosaic.quantize(colors=256 - len(THEME_COLORS)).getpalette()
    colours = automatic[: 3 * (256 - len(THEME_COLORS))]
    for hex_colour in THEME_COLORS:
        colours += [int(hex_colour[i : i + 2], 16) for i in (1, 3, 5)]
    palette = Image.new("P", (1, 1))
    palette.putpalette(colours)
    images = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    images[0].save(path, save_all=True, append_images=images[1:], duration=1000 // fps,
                   loop=0, optimize=True)
