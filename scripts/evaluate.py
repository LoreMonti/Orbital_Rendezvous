"""Evaluate the trained agent against the classical references, on the same starts.

Flies the agent (deterministic policy, no exploration noise) and a sweep of LQR
controllers over the same seeded initial conditions, none of which was seen in
training, and computes the ideal two-impulse delta-v for each start. Prints a
table, and saves a plot of delta-v against time to dock: the LQR sweep traces
the trade-off between the two, and the agent is a single point on it.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --watch 5       # also replay 5 agent attempts
    python scripts/evaluate.py --episodes 50   # quicker
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from orbital_rendezvous import RendezvousEnv
from orbital_rendezvous.baselines import LQRController, best_two_impulse
from orbital_rendezvous.evaluation import Summary, evaluate, pareto_front, summarise
from orbital_rendezvous.utils import build_configs, load_config

# Held-out starts: training seeds its environments from 0 upwards.
FIRST_SEED = 10_000
APPROACH_TIMES = (50.0, 75.0, 100.0, 150.0, 200.0, 300.0)
FUEL_WEIGHTS = tuple(np.logspace(-4, 0, 9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/ppo_default.yaml")
    parser.add_argument("--model", default="models/ppo_rendezvous.zip")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--plot", default="assets/delta_v_vs_time.png")
    parser.add_argument("--results", default="assets/evaluation.json")
    parser.add_argument(
        "--watch", type=int, default=0, help="Replay this many agent attempts in the window."
    )
    return parser.parse_args()


def row(name: str, s) -> str:
    return (
        f"{name:<34s} {100 * s.success_rate:6.1f} %  {s.delta_v_median:6.2f} "
        f"({s.delta_v_p5:4.2f}-{s.delta_v_p95:4.2f})  {s.time_median:6.0f} s  "
        f"{100 * s.docking_speed_median:5.1f} cm/s"
    )


def main() -> None:
    args = parse_args()
    env_config, reward_config = build_configs(load_config(args.config))
    env = RendezvousEnv(env_config, reward_config)
    seeds = range(FIRST_SEED, FIRST_SEED + args.episodes)

    model = PPO.load(args.model)
    agent_runs = evaluate(env, lambda e, obs: model.predict(obs, deterministic=True)[0], seeds)
    agent = summarise(agent_runs)

    print(f"Sweeping {len(APPROACH_TIMES) * len(FUEL_WEIGHTS)} LQR controllers...")
    sweep = []
    for tau in APPROACH_TIMES:
        for weight in FUEL_WEIGHTS:
            lqr = LQRController.from_env(env, tau, weight)
            summary = summarise(evaluate(env, lambda e, obs, c=lqr: c.act(e.state), seeds))
            sweep.append({"approach_time": tau, "fuel_weight": float(weight), **asdict(summary)})

    # Only controllers that dock every time compete on cost with an agent that does.
    reliable = [s for s in sweep if s["success_rate"] == 1.0]
    front = [reliable[i] for i in pareto_front(
        [(s["time_median"], s["delta_v_median"]) for s in reliable]
    )]
    # The front is sorted by time: its ends are the fastest and the cheapest
    # LQR tunings that still dock every time.
    fastest, cheapest = (front[0], front[-1]) if front else (None, None)

    horizon = env_config.max_episode_steps * env_config.time_step
    durations = np.arange(env_config.time_step, horizon + 1, env_config.time_step)
    impulses = []
    for seed in seeds:
        env.reset(seed=seed)
        impulses.append(best_two_impulse(env.state, env.n, durations))
    ideal_dv = np.array([dv for dv, _ in impulses])
    ideal_t = np.array([t for _, t in impulses])

    print()
    header = "docked   delta-v m/s (5-95 %)   time   dock speed"
    print(f"{f'{args.episodes} unseen starts':<34s}  {header}")
    print(row("PPO agent (deterministic)", agent))
    for label, s in (("LQR, fastest that always docks", fastest),
                     ("LQR, cheapest that always docks", cheapest)):
        if s is not None:
            print(row(label, Summary(**{k: s[k] for k in asdict(agent)})))
            tuning = f"tau = {s['approach_time']:.0f} s, fuel weight = {s['fuel_weight']:.0e}"
            print(f"{'':<34s} {tuning}")
    crashes = max(s["outcomes"].get("crashed", 0) for s in sweep) / args.episodes
    if crashes:
        print(f"{'':<34s} more aggressive LQR tunings crash in up to {100 * crashes:.0f} %")
    print(f"{'ideal two-impulse (not flyable)':<34s} {'':8s}  {np.median(ideal_dv):6.2f} "
          f"({np.percentile(ideal_dv, 5):4.2f}-{np.percentile(ideal_dv, 95):4.2f})  "
          f"{np.median(ideal_t):6.0f} s")

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "w") as handle:
        json.dump({
            "episodes": args.episodes,
            "agent": asdict(agent),
            "lqr_sweep": sweep,
            "lqr_front": front,
            "two_impulse": {"delta_v_median": float(np.median(ideal_dv)),
                            "duration_median": float(np.median(ideal_t))},
        }, handle, indent=2)

    plot(args.plot, agent, sweep, front, float(np.median(ideal_dv)), args.episodes)
    print(f"\nPlot saved to {args.plot}, numbers to {args.results}")

    if args.watch:
        from orbital_rendezvous.live_view import LiveView

        view = LiveView.from_env(env, replay_frames=90)
        for k, run in enumerate(agent_runs[: args.watch]):
            view.show_episode(run.positions, run.velocities, run.thrusts, run.outcome, k + 1)
        view.plt.show(block=True)


def plot(path, agent, sweep, front, ideal_dv, episodes) -> None:
    import matplotlib.pyplot as plt

    from orbital_rendezvous.live_view import BLUE, GREEN, GRID, MUTED, PANEL, TEXT

    fig, ax = plt.subplots(figsize=(8.0, 5.2), facecolor=PANEL)
    ax.set_facecolor(PANEL)
    reliable = [s for s in sweep if s["success_rate"] == 1.0]
    ax.scatter([s["time_median"] for s in reliable], [s["delta_v_median"] for s in reliable],
               s=14, color=MUTED, alpha=0.6, label="LQR, one tuning each (docks every time)")
    ax.plot([s["time_median"] for s in front], [s["delta_v_median"] for s in front],
            color=BLUE, lw=2, marker="o", ms=4, label="LQR, best trade-offs")
    ax.scatter([agent.time_median], [agent.delta_v_median], s=220, marker="*", color=GREEN,
               zorder=5, label="PPO agent")
    ax.axhline(ideal_dv, color=TEXT, ls="--", lw=1,
               label="ideal two-impulse transfer, best duration (not flyable)")
    ax.set_xlabel("time to dock, median [s]", color=TEXT)
    ax.set_ylabel(r"fuel spent, median $\Delta v$ [m/s]", color=TEXT)
    ax.set_title(f"Fuel against time on {episodes} unseen starts  (lower left is better)",
                 color=TEXT, fontsize=11)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(alpha=0.25, color=GRID)
    ax.set_ylim(bottom=0)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PANEL)
    plt.close(fig)


if __name__ == "__main__":
    main()
