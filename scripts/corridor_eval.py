"""Evaluate agents on the oriented target against the V-bar procedure, on the same starts.

Flies each trained agent (deterministic policy) and the V-bar procedure, with
two time constants of its LQR, from the same 200 held-out starts in every
direction, with the full rule: keep-out sphere of 20 m, approach cone of 15
degrees. Prints the dockings, the keep-out violations, the median cost of the
dockings, and the dockings by direction of the start, measured from the
docking axis: from beyond 90 degrees the chaser must go around the station.

An agent that docks only from some starts would look cheap or fast next to a
procedure that also flies the hard ones, so each agent's costs are then set
against the V-bar procedure's on the very starts that agent docked. Saves both
tables as JSON.

Usage:
    python scripts/corridor_eval.py --models models/corridor_seed0.zip models/corridor_seed1.zip
    python scripts/corridor_eval.py --models models/corridor_seed*.zip --episodes 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from orbital_rendezvous import RendezvousEnv
from orbital_rendezvous.baselines import VbarApproach
from orbital_rendezvous.evaluation import HELD_OUT_SEED, docked_by_sector, evaluate
from orbital_rendezvous.rewards import Outcome
from orbital_rendezvous.utils import build_configs, load_config

SECTORS = (0.0, 45.0, 90.0, 135.0, 180.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/ppo_corridor.yaml")
    parser.add_argument("--models", nargs="*", default=[], help="Trained agents to evaluate.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--results", default="assets/corridor_evaluation.json")
    return parser.parse_args()


def summary(runs) -> dict:
    docked = [r for r in runs if r.outcome is Outcome.DOCKED]
    return {
        "docked": len(docked),
        "attempts": len(runs),
        "violations": sum(r.violated for r in runs),
        "delta_v_median": float(np.median([r.delta_v for r in docked])) if docked else None,
        "time_median": float(np.median([r.time for r in docked])) if docked else None,
        "docked_by_sector": docked_by_sector(runs, SECTORS),
    }


def row(name: str, s: dict) -> str:
    cost = (f"{s['delta_v_median']:5.2f} m/s {s['time_median']:6.0f} s"
            if s["docked"] else f"{'-':>9s} {'-':>8s}")
    sectors = "  ".join(f"{d:>3d}/{n:<3d}" for d, n in s["docked_by_sector"])
    return (f"{name:<28s} {s['docked']:>3d}/{s['attempts']:<3d} {s['violations']:>5d}  "
            f"{cost}   {sectors}")


def main() -> None:
    args = parse_args()
    env_config, reward_config = build_configs(load_config(args.config))
    env = RendezvousEnv(env_config, reward_config)
    seeds = range(HELD_OUT_SEED, HELD_OUT_SEED + args.episodes)

    results, vbar, agents = {}, {}, {}
    for tau in (100.0, 200.0):
        name = f"V-bar procedure, tau = {tau:.0f} s"
        vbar[name] = evaluate(env, VbarApproach(env, approach_time=tau), seeds)
        results[name] = summary(vbar[name])
    for path in args.models:
        model = PPO.load(path)
        runs = evaluate(env, lambda e, obs, m=model: m.predict(obs, deterministic=True)[0], seeds)
        agents[Path(path).stem] = runs
        results[Path(path).stem] = summary(runs)

    edges = zip(SECTORS[:-1], SECTORS[1:], strict=True)
    labels = "  ".join(f"{f'{a:.0f}-{b:.0f}°':>7s}" for a, b in edges)
    print(f"{'':<56s}   docked, by start angle from the docking axis")
    print(f"{'controller':<28s} {'docked':>7s} {'viol.':>5s}  {'Δv':>9s} {'time':>8s}   {labels}")
    for name, s in results.items():
        print(row(name, s))

    same_starts = {}
    if agents:
        print("\nOn the starts each agent docks: median delta-v and time, and how often the "
              "agent beats the procedure start by start")
    for name, runs in agents.items():
        docked = [i for i, r in enumerate(runs) if r.outcome is Outcome.DOCKED]
        if not docked:
            continue
        entry = {"starts": len(docked), "agent": {
            "delta_v_median": float(np.median([runs[i].delta_v for i in docked])),
            "time_median": float(np.median([runs[i].time for i in docked])),
        }}
        agent = entry["agent"]
        line = (f"{name:<16s} {len(docked):>3d} starts: agent "
                f"{agent['delta_v_median']:4.2f} m/s {agent['time_median']:5.0f} s")
        for reference, reference_runs in vbar.items():
            cheaper = np.mean([runs[i].delta_v < reference_runs[i].delta_v for i in docked])
            faster = np.mean([runs[i].time < reference_runs[i].time for i in docked])
            entry[reference] = {
                "delta_v_median": float(np.median([reference_runs[i].delta_v for i in docked])),
                "time_median": float(np.median([reference_runs[i].time for i in docked])),
                "agent_cheaper": float(cheaper),
                "agent_faster": float(faster),
            }
            line += (f" | tau {reference.split('= ')[1]}: "
                     f"{entry[reference]['delta_v_median']:4.2f} m/s "
                     f"{entry[reference]['time_median']:5.0f} s, agent cheaper "
                     f"{100 * cheaper:3.0f} %, faster {100 * faster:3.0f} %")
        same_starts[name] = entry
        print(line)

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps({
        "description": "Oriented target: keep-out sphere of 20 m, approach cone of 15 degrees. "
                       f"{args.episodes} held-out starts in every direction; costs are medians "
                       "over the dockings; sectors are start angles from the docking axis.",
        "sectors_deg": list(SECTORS),
        "results": results,
        "same_starts": same_starts,
    }, indent=1))
    print(f"Results saved to {args.results}")


if __name__ == "__main__":
    main()
