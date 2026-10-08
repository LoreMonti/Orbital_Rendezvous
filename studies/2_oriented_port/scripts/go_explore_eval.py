"""Evaluate Go-Explore as a planner on the 200 held-out starts of the oriented target (Step 27).

Flies `go_explore.GoExplorePlanner` from the same starts as `corridor_eval.py`
and `plan_eval.py`, in parallel over the starts, and prints the dockings, the
violations, the median cost of the dockings and the dockings by direction of
the start. With ``--replan-every 0`` the planner explores once, at the start,
and flies that plan to the end, open loop.

With ``--thrust-error`` the thrust applied differs from the one commanded, a
fresh error every step: its magnitude is off by a Gaussian fraction of that
standard deviation, and its direction by a Gaussian angle of
``--angle-error`` degrees. The planner does not know it; it only sees where
the chaser ends up, which is what replanning is for.

Usage:
    python studies/2_oriented_port/scripts/go_explore_eval.py --replan-every 30
    python studies/2_oriented_port/scripts/go_explore_eval.py \
        --replan-every 0 --thrust-error 0.1 --angle-error 3
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from orbital_rendezvous import RendezvousEnv
from orbital_rendezvous.core.evaluation import HELD_OUT_SEED, docked_by_sector, rollout
from orbital_rendezvous.core.rewards import Outcome
from orbital_rendezvous.core.utils import build_configs, load_config
from orbital_rendezvous.planning.go_explore import GoExploreConfig, GoExplorePlanner

SECTORS = (0.0, 45.0, 90.0, 135.0, 180.0)
EDGES = list(zip(SECTORS[:-1], SECTORS[1:], strict=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="studies/2_oriented_port/configs/ppo_corridor.yaml")
    parser.add_argument("--replan-every", type=int, default=30, help="steps; 0 plans once")
    parser.add_argument("--thrust-error", type=float, default=0.0, help="std of magnitude error")
    parser.add_argument("--angle-error", type=float, default=0.0,
                        help="std of direction error, deg")
    parser.add_argument("--replan-margin", type=int, default=2_000,
                        help="rounds a replan keeps searching after its first docking")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--results",
                        default="studies/2_oriented_port/assets/go_explore_planner.json")
    return parser.parse_args()


class ThrustError:
    """A controller whose commanded thrust is applied with a random error."""

    def __init__(self, controller, magnitude: float, angle_deg: float, seed: int):
        self.controller, self.magnitude = controller, magnitude
        self.angle = np.radians(angle_deg)
        self.rng = np.random.default_rng(seed)

    def __call__(self, env, obs):
        action = np.asarray(self.controller(env, obs), dtype=float)
        scale = 1.0 + self.magnitude * self.rng.normal()
        turn = self.angle * self.rng.normal()
        c, s = np.cos(turn), np.sin(turn)
        return np.clip(scale * np.array([c * action[0] - s * action[1],
                                         s * action[0] + c * action[1]]), -1.0, 1.0)


def fly(job):
    config_path, seed, replan_every, magnitude, angle, margin = job
    env_config, reward_config = build_configs(load_config(config_path))
    env = RendezvousEnv(env_config, reward_config)
    every = replan_every if replan_every > 0 else 10**9
    planner = GoExplorePlanner(env, GoExploreConfig(), replan_every=every, seed=seed,
                               replan_after_docking=margin)
    controller = ThrustError(planner, magnitude, angle, seed) if magnitude or angle else planner
    began = time.perf_counter()
    with np.errstate(all="ignore"):   # spurious matmul warnings of Accelerate on macOS
        run = rollout(env, controller, seed)
    return run, planner.searches, time.perf_counter() - began


def main() -> None:
    args = parse_args()
    seeds = range(HELD_OUT_SEED, HELD_OUT_SEED + args.episodes)
    jobs = [(args.config, s, args.replan_every, args.thrust_error, args.angle_error,
             args.replan_margin) for s in seeds]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(fly, jobs))
    runs = [r for r, _, _ in results]
    docked = [r for r in runs if r.outcome is Outcome.DOCKED]
    sectors = docked_by_sector(runs, SECTORS)
    outcomes = {o.value: sum(r.outcome is o for r in runs) for o in {r.outcome for r in runs}}
    dv = float(np.median([r.delta_v for r in docked])) if docked else float("nan")
    t = float(np.median([r.time for r in docked])) if docked else float("nan")
    name = (f"replan every {args.replan_every}" if args.replan_every else "open loop") + (
        f", thrust error {args.thrust_error:.2f}, {args.angle_error:.0f} deg"
        if args.thrust_error or args.angle_error else "") + (
        f", replan margin {args.replan_margin}" if args.replan_every else "")
    print(f"{name}: {len(docked)}/{len(runs)} docked, {outcomes}, {dv:.2f} m/s, {t:.0f} s")
    print("by sector:", sectors)
    print(f"searches per flight, median {np.median([n for _, n, _ in results]):.0f}; "
          f"minutes per flight, median {np.median([m for _, _, m in results]) / 60:.1f}")
    path = Path(args.results)
    table = json.loads(path.read_text()) if path.exists() else {"results": {}}
    table["description"] = ("Go-Explore as a planner on the 200 held-out starts of the "
                            "oriented target; thrust errors as in "
                            "studies/2_oriented_port/scripts/go_explore_eval.py.")
    table["results"][name] = {
        "docked": len(docked), "episodes": len(runs), "outcomes": outcomes,
        "delta_v_median": dv, "time_median": t,
        "by_sector": [{"from_deg": a, "to_deg": b, "docked": d, "starts": n}
                      for (a, b), (d, n) in zip(EDGES, sectors, strict=True)],
        "searches_median": float(np.median([n for _, n, _ in results])),
        "minutes_per_flight_median": float(np.median([m for _, _, m in results]) / 60),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=1))
    print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
