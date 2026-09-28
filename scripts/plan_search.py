"""Do more waypoints pay? A beam search over plans of up to three waypoints.

Before teaching a planner to choose how many waypoints to fly (Step 18), this
measures what more of them could win. From each of the 200 held-out starts, a
beam search (`hierarchy.beam_search`) flies plans of up to three waypoints
from the menu with the frozen pilots, and keeps the cheapest that docks with
at most one, two and three waypoints, on the cost of the configuration,
``J = fuel_weight * delta_v + time_weight * T``. In parallel, about ten
minutes on four processes.

Usage:
    python scripts/plan_search.py
    python scripts/plan_search.py --episodes 50 --jobs 8
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from orbital_rendezvous.evaluation import HELD_OUT_SEED, start_angle
from orbital_rendezvous.hierarchy import beam_search, best_plan
from orbital_rendezvous.utils import load_config, make_env

_ENV = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/ppo_planner_relative.yaml",
                        help="Pilots, menu and the weights of the cost J.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--results", default="assets/plan_search.json")
    return parser.parse_args()


def _init(config_path: str) -> None:
    global _ENV
    _ENV = make_env(load_config(config_path))


def _search(task: tuple[int, int, int]) -> dict:
    seed, width, depth = task
    flown = beam_search(_ENV, seed, width, depth)
    _ENV.reset(seed=seed)
    row = {"seed": seed, "angle": start_angle(_ENV.state[:2]), "plans_flown": len(flown)}
    for n in range(1, depth + 1):
        plan, value = best_plan(flown, n)
        row[f"best_{n}"] = {"plan": [list(w) for w in plan], **value}
    return row


def main() -> None:
    args = parse_args()
    seeds = range(HELD_OUT_SEED, HELD_OUT_SEED + args.episodes)
    tasks = [(s, args.width, args.depth) for s in seeds]
    with Pool(args.jobs, initializer=_init, initargs=(args.config,)) as pool:
        rows = pool.map(_search, tasks, chunksize=5)

    print(f"{'best plan with at most':<24s} {'docked':>7s} {'J':>7s} {'Δv':>9s} {'time':>7s}   "
          f"plans with 0, 1, 2, 3 waypoints")
    for n in range(1, args.depth + 1):
        best = [r[f"best_{n}"] for r in rows]
        lengths = np.bincount([len(b["plan"]) for b in best], minlength=args.depth + 1)
        print(f"{n} waypoint(s){'':<11s} {sum(b['docked'] for b in best):>3d}/{len(best):<3d} "
              f"{np.median([b['cost'] for b in best]):7.2f} "
              f"{np.median([b['delta_v'] for b in best]):5.3f} m/s "
              f"{np.median([b['time'] for b in best]):5.0f} s   {lengths.tolist()}")
    gain = np.array([r["best_1"]["cost"] - r[f"best_{args.depth}"]["cost"] for r in rows])
    behind = np.array([r["angle"] > 135.0 for r in rows])
    print(f"cost saved by {args.depth} waypoints over 1: mean {gain.mean():.3f}, largest "
          f"{gain.max():.3f}, starts saving more than 0.5: {(gain > 0.5).sum()}; behind the "
          f"station ({behind.sum()} starts): largest {gain[behind].max():.3f}")

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps({
        "description": f"Beam search, width {args.width}, plans of up to {args.depth} waypoints "
                       f"from the menu, flown by the frozen pilots from {args.episodes} held-out "
                       "starts; the cheapest plan that docks with at most n waypoints.",
        "rows": rows,
    }, indent=1))
    print(f"Results saved to {args.results}")


if __name__ == "__main__":
    main()
