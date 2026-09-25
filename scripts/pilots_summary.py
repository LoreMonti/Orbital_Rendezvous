"""The two learned pilots on the oriented target, in one figure.

Left: a few approaches flown by the pilots and by the V-bar procedure from the
same held-out starts, chosen around the station, the one behind it included,
with the keep-out sphere, the approach cone and the hold point. Right: median
delta-v against median time to dock on the 200 held-out starts, for every
pairing of trained pilots and for the two tunings of the procedure, read from
the results of `corridor_eval.py`.

Usage:
    python scripts/pilots_summary.py
    python scripts/pilots_summary.py --go-to models/goto_seed1_best.zip \\
        --final models/final_approach_seed1_best.zip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/ppo_corridor.yaml")
    parser.add_argument("--go-to", default="models/goto_seed1_best.zip")
    parser.add_argument("--final", default="models/final_approach_seed1_best.zip")
    parser.add_argument("--pilot-configs", nargs=2,
                        default=["configs/ppo_goto.yaml", "configs/ppo_final_approach.yaml"])
    parser.add_argument("--angles", nargs="*", type=float, default=[178.0, 150.0, 115.0, 30.0],
                        help="Start angles from the docking axis of the approaches drawn.")
    parser.add_argument("--results", default="assets/pilots_evaluation.json")
    parser.add_argument("--plot", default="assets/pilots.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stable_baselines3 import PPO

    from orbital_rendezvous.baselines import VbarApproach
    from orbital_rendezvous.evaluation import HELD_OUT_SEED, rollout, start_angle
    from orbital_rendezvous.game_view import AMBER, BLUE, GREEN, GRID, MUTED, PANEL, RED, TEXT
    from orbital_rendezvous.hierarchy import HierarchicalPilot
    from orbital_rendezvous.utils import build_configs, load_config, make_env

    env = make_env(load_config(args.config))
    cfg = env.config
    configs = [build_configs(load_config(c))[0] for c in args.pilot_configs]
    pilot = HierarchicalPilot.from_models(PPO.load(args.go_to), PPO.load(args.final), *configs,
                                          keep_out=cfg.keep_out_radius)
    vbar = VbarApproach(env, approach_time=200.0)

    # The held-out starts closest to the angles asked for.
    angles = {}
    for seed in range(HELD_OUT_SEED, HELD_OUT_SEED + 200):
        env.reset(seed=seed)
        angles[seed] = start_angle(env.state[:2])
    chosen = [min(angles, key=lambda s, a=a: abs(angles[s] - a)) for a in args.angles]

    fig, (path_ax, cost_ax) = plt.subplots(
        1, 2, figsize=(13.0, 5.8), facecolor=PANEL, gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    for ax in (path_ax, cost_ax):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(GRID)

    # Left: trajectories, the V-bar (along-track, y) across and the radial x up.
    radius, cone = cfg.keep_out_radius, np.radians(cfg.approach_cone_deg)
    circle = np.linspace(0.0, 2.0 * np.pi, 200)
    path_ax.fill(radius * np.cos(circle), radius * np.sin(circle), color=RED, alpha=0.12, lw=0)
    path_ax.plot(radius * np.cos(circle), radius * np.sin(circle), color=RED, lw=1.0)
    for sign in (1, -1):
        path_ax.plot([0, 60 * np.cos(cone)], [0, sign * 60 * np.sin(cone)], color=MUTED, lw=0.8,
                     ls=":")
    path_ax.plot(30, 0, "o", color=TEXT, ms=4)
    path_ax.annotate("hold point", (30, 0), xytext=(58, 30), color=TEXT, fontsize=8,
                     arrowprops={"arrowstyle": "-", "color": TEXT, "lw": 0.6})
    path_ax.plot(0, 0, "o", color=TEXT, ms=3)
    path_ax.annotate("port", (0, 0), xytext=(-6, -3), textcoords="offset points", color=TEXT,
                     fontsize=8, ha="right")
    path_ax.annotate(f"keep-out sphere, {radius:.0f} m", (0, -radius), xytext=(0, -12),
                     textcoords="offset points", color=RED, fontsize=8, ha="center")
    for seed in chosen:
        for controller, colour, width in ((vbar, MUTED, 1.2), (pilot, GREEN, 2.0)):
            run = rollout(env, controller, seed)
            track = np.vstack([run.start[:2], run.positions])
            path_ax.plot(track[:, 1], track[:, 0], color=colour, lw=width)
        path_ax.plot(run.start[1], run.start[0], "o", color=AMBER, ms=5)
        path_ax.annotate(f"{angles[seed]:.0f}°", (run.start[1], run.start[0]), xytext=(6, 4),
                         textcoords="offset points", color=AMBER, fontsize=9)
    path_ax.plot([], [], color=GREEN, lw=2.0, label="two learned pilots")
    path_ax.plot([], [], color=MUTED, lw=1.2, label="V-bar procedure")
    path_ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9,
                   loc="upper right")
    path_ax.set_aspect("equal")
    path_ax.set_xlabel("along-track y, the V-bar [m]", color=TEXT)
    path_ax.set_ylabel("radial x [m]", color=TEXT)
    path_ax.set_title("Around the station and in through the port", color=TEXT, fontsize=11,
                      loc="left")

    # Right: the cost of every pairing, against the procedure.
    results = json.loads(Path(args.results).read_text())["results"]
    vbar_points = [(r["time_median"], r["delta_v_median"]) for name, r in results.items()
                   if name.startswith("V-bar")]
    xs, ys = zip(*sorted(vbar_points), strict=True)
    # Drawn over the pilots' points, which can land right on it.
    cost_ax.plot(xs, ys, color=MUTED, ls="--", lw=1.2, marker="s", ms=7, zorder=3,
                 markeredgecolor=PANEL)
    for (x, y), tau in zip(sorted(vbar_points), (100, 200), strict=True):
        cost_ax.annotate(f"V-bar, τ = {tau} s", (x, y), xytext=(8, 4 if tau == 100 else -14),
                         textcoords="offset points", color=MUTED, fontsize=9)
    colours = (AMBER, GREEN, BLUE)
    for name, r in results.items():
        if " + " not in name:
            continue
        go_to_seed = int(name.split("seed")[1][0])
        cost_ax.plot(r["time_median"], r["delta_v_median"], "o", ms=7,
                     color=colours[go_to_seed % len(colours)])
    for seed, colour in enumerate(colours):
        points = [(r["time_median"], r["delta_v_median"]) for name, r in results.items()
                  if name.startswith(f"goto_seed{seed}")]
        if points:
            x, y = max(points)
            cost_ax.annotate(f"go-to seed {seed}", (x, y), xytext=(8, -3),
                             textcoords="offset points", color=colour, fontsize=9)
    cost_ax.grid(alpha=0.25, color=GRID)
    cost_ax.set_xlabel("time to dock, median [s]", color=TEXT)
    cost_ax.set_ylabel("Δv, median [m/s]", color=TEXT)
    cost_ax.set_title("Every pairing docks 200 / 200: the cost of each", color=TEXT,
                      fontsize=11, loc="left")
    cost_ax.set_xlim(min(xs + tuple(r["time_median"] for r in results.values())) - 100,
                     max(xs + tuple(r["time_median"] for r in results.values())) + 350)

    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=130, bbox_inches="tight", facecolor=PANEL)
    plt.close(fig)
    print(f"Plot saved to {args.plot}")


if __name__ == "__main__":
    main()
