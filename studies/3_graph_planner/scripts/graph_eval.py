"""Evaluate the graph pilot on the 200 held-out starts of the oriented target (Step 28).

Builds the graph of exact manoeuvres once, then flies `graph.GraphPilot` from
the same starts as the evaluations of Part 2, in parallel over the starts, and
prints the dockings, the violations, the median cost of the dockings, the
dockings by direction of the start and the computing time per decision. With
``--thrust-error`` and ``--angle-error`` the thrust applied differs from the one
commanded, as in Part 2's `go_explore_eval.py`. Results are added to a JSON file.

Usage:
    python studies/3_graph_planner/scripts/graph_eval.py
    python studies/3_graph_planner/scripts/graph_eval.py --thrust-error 0.1 --angle-error 3
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
from orbital_rendezvous.planning.graph import CWGraph, GraphPilot

SECTORS = (0.0, 45.0, 90.0, 135.0, 180.0)
EDGES = list(zip(SECTORS[:-1], SECTORS[1:], strict=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="studies/2_oriented_port/configs/ppo_corridor.yaml")
    parser.add_argument("--thrust-error", type=float, default=0.0, help="std of magnitude error")
    parser.add_argument("--angle-error", type=float, default=0.0,
                        help="std of direction error, deg")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--results", default="studies/3_graph_planner/assets/graph_pilot.json")
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
    config_path, seed, magnitude, angle = job
    env_config, reward_config = build_configs(load_config(config_path))
    env = RendezvousEnv(env_config, reward_config)
    pilot = GraphPilot(env, CWGraph(RendezvousEnv(env_config, reward_config)), seed=seed)
    controller = ThrustError(pilot, magnitude, angle, seed) if magnitude or angle else pilot
    began = time.perf_counter()
    with np.errstate(all="ignore"):   # spurious matmul warnings of Accelerate on macOS
        run = rollout(env, controller, seed)
    return run, (time.perf_counter() - began) / len(run.positions)


def main() -> None:
    args = parse_args()
    seeds = range(HELD_OUT_SEED, HELD_OUT_SEED + args.episodes)
    jobs = [(args.config, s, args.thrust_error, args.angle_error) for s in seeds]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(fly, jobs))
    runs = [r for r, _ in results]
    docked = [r for r in runs if r.outcome is Outcome.DOCKED]
    sectors = docked_by_sector(runs, SECTORS)
    outcomes = {o.value: sum(r.outcome is o for r in runs) for o in {r.outcome for r in runs}}
    dv = float(np.median([r.delta_v for r in docked])) if docked else float("nan")
    t = float(np.median([r.time for r in docked])) if docked else float("nan")
    ms = 1000 * float(np.median([s for _, s in results]))
    name = "graph pilot" + (f", thrust error {args.thrust_error:.2f}, {args.angle_error:.0f} deg"
                            if args.thrust_error or args.angle_error else "")
    print(f"{name}: {len(docked)}/{len(runs)} docked, {outcomes}, {dv:.2f} m/s, {t:.0f} s, "
          f"{ms:.0f} ms per decision")
    print("by sector:", sectors)
    path = Path(args.results)
    table = json.loads(path.read_text()) if path.exists() else {"results": {}}
    table["description"] = ("The graph pilot of Step 28 on the 200 held-out starts of the "
                            "oriented target; thrust errors as in graph_eval.py.")
    table["results"][name] = {
        "docked": len(docked), "episodes": len(runs), "outcomes": outcomes,
        "delta_v_median": dv, "time_median": t, "ms_per_decision": ms,
        "by_sector": [{"from_deg": a, "to_deg": b, "docked": d, "starts": n}
                      for (a, b), (d, n) in zip(EDGES, sectors, strict=True)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=1))
    print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
