"""Watch the agent and an LQR controller fly the same approach, side by side.

Both start from the same point, one of the held-out starts of `evaluate.py`,
and are drawn in the same game view at the same scale and the same clock, so
the status bars compare them directly: which one docks first, and which one
burns less fuel. The LQR tuning is read from the results of `evaluate.py`:
by default the fastest one that always docks, the agent's closest rival.

Usage:
    python scripts/play.py                        # three starts, in a window
    python scripts/play.py --lqr cheapest         # against the patient LQR
    python scripts/play.py --start 7 --starts 1   # one chosen start
    python scripts/play.py --gif assets/side_by_side.gif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from orbital_rendezvous import RendezvousEnv
from orbital_rendezvous.baselines import LQRController
from orbital_rendezvous.evaluation import HELD_OUT_SEED, rollout
from orbital_rendezvous.game_view import BACKGROUND, GameView, add_legend, scene_extent
from orbital_rendezvous.utils import build_configs, load_config

# Used when evaluate.py has not been run: the fastest LQR that always docked
# on the default configuration.
FALLBACK_TUNING = {"approach_time": 100.0, "fuel_weight": 1e-3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/ppo_default.yaml")
    parser.add_argument("--model", default="models/ppo_rendezvous.zip")
    parser.add_argument("--results", default="assets/evaluation.json")
    parser.add_argument("--lqr", choices=("fastest", "cheapest"), default="fastest")
    parser.add_argument("--start", type=int, default=0, help="First held-out start to fly.")
    parser.add_argument("--starts", type=int, default=3, help="How many starts, one after another.")
    parser.add_argument("--frames", type=int, default=120, help="Frames per start.")
    parser.add_argument("--gif", default=None, help="Save a GIF here instead of opening a window.")
    return parser.parse_args()


def lqr_tuning(results: str, which: str) -> dict[str, float]:
    path = Path(results)
    if not path.exists():
        print(f"{results} not found, using the default tuning: run evaluate.py for the real one")
        return FALLBACK_TUNING
    front = json.loads(path.read_text())["lqr_front"]
    return front[0] if which == "fastest" else front[-1]


def main() -> None:
    args = parse_args()
    if args.gif:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    env_config, reward_config = build_configs(load_config(args.config))
    env = RendezvousEnv(env_config, reward_config)
    model = PPO.load(args.model)
    tuning = lqr_tuning(args.results, args.lqr)
    lqr = LQRController.from_env(env, tuning["approach_time"], tuning["fuel_weight"])

    def agent(e, obs):
        return model.predict(obs, deterministic=True)[0]

    def classical(e, obs):
        return lqr.act(e.state)

    fig = plt.figure(figsize=(15.0, 8.4), facecolor=BACKGROUND)
    fig.canvas.manager.set_window_title("Orbital Rendezvous: agent against LQR")
    grid = fig.add_gridspec(2, 2, height_ratios=[10.0, 1.2], wspace=0.08, hspace=0.12,
                            left=0.02, right=0.98, top=0.93, bottom=0.02)
    views = [GameView.from_env(fig, grid[0, i], env) for i in range(2)]
    add_legend(fig.add_subplot(grid[1, :]), ncol=6)
    names = (
        "PPO AGENT",
        f"LQR, {args.lqr} that always docks (tau {tuning['approach_time']:.0f} s)",
    )

    # One entry per frame: (start index, step), with a pause after each start.
    starts = [HELD_OUT_SEED + args.start + i for i in range(args.starts)]
    runs = [(rollout(env, agent, seed), rollout(env, classical, seed)) for seed in starts]
    hold = args.frames // 4
    schedule = []
    for i, pair in enumerate(runs):
        last = max(run.positions.shape[0] for run in pair) - 1
        steps = np.unique(np.linspace(0, last, args.frames).astype(int))
        schedule += [(i, int(k)) for k in steps] + [(i, last)] * hold

    loaded = [-1]

    def update(frame):
        i, k = schedule[frame]
        if loaded[0] != i:
            extent = scene_extent(*(run.positions for run in runs[i]))
            for view, run, name in zip(views, runs[i], names, strict=True):
                view.load(run.positions, run.velocities, run.thrusts, run.outcome,
                          f"{name}  ·  start {args.start + i + 1}", extent=extent)
            loaded[0] = i
        for view in views:
            view.draw(k)
        return []

    for pair, seed in zip(runs, starts, strict=True):
        agent_run, lqr_run = pair
        print(f"start {seed - HELD_OUT_SEED + 1}: "
              f"agent {agent_run.outcome.value} in {agent_run.time:.0f} s, "
              f"{agent_run.delta_v:.2f} m/s  |  "
              f"LQR {lqr_run.outcome.value} in {lqr_run.time:.0f} s, {lqr_run.delta_v:.2f} m/s")

    animation = FuncAnimation(fig, update, frames=len(schedule), interval=40, blit=False,
                              repeat=False)
    if args.gif:
        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        # A lower resolution keeps the file small enough for a README.
        animation.save(args.gif, writer=PillowWriter(fps=20), dpi=72,
                       savefig_kwargs={"facecolor": BACKGROUND})
        print(f"GIF saved to {args.gif}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
