"""Evaluate the sampling planner on the oriented target, on the 200 held-out starts.

Flies `planning.SamplingPlanner` with each requested setting from the same
starts as `corridor_eval.py`, in parallel over the starts, and prints the
dockings, the violations, the median cost of the dockings, the dockings by
direction of the start, and the planning time per decision. Saves the table
as JSON.

Each setting is ``value:horizon:block``, for example ``potential:20:4``: the
value beyond the horizon (none, distance, potential, or learned with
``--value-net``), the horizon in steps, and the steps a sampled thrust is held.

Usage:
    python scripts/plan_eval.py --settings distance:50:5 potential:20:4
    python scripts/plan_eval.py --settings potential:10:2 --episodes 50 --workers 4
    python scripts/plan_eval.py --settings learned:20:4 --value-net runs/value_loop/seed0
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from orbital_rendezvous import RendezvousEnv
from orbital_rendezvous.evaluation import HELD_OUT_SEED, docked_by_sector, rollout
from orbital_rendezvous.planning import PlannerConfig, SamplingPlanner
from orbital_rendezvous.rewards import Outcome
from orbital_rendezvous.utils import build_configs, load_config
from orbital_rendezvous.value import LearnedValue, ValueNet

SECTORS = (0.0, 45.0, 90.0, 135.0, 180.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/ppo_corridor.yaml")
    parser.add_argument("--settings", nargs="+", default=["distance:50:5", "potential:20:4"])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--init-std", type=float, default=0.3)
    parser.add_argument("--results", default="assets/planner_evaluation.json")
    parser.add_argument("--value-net", default=None,
                        help="run directory of value_loop.py, with its value_*.npz networks, "
                             "for settings learned:H:B")
    return parser.parse_args()


def planner_config(setting: str, init_std: float) -> PlannerConfig:
    value, horizon, block = setting.split(":")
    return PlannerConfig(value=value, horizon=int(horizon), block=int(block), init_std=init_std)


def fly(job: tuple[str, PlannerConfig, int, str | None]):
    """One start, flown by a fresh planner: the rollout and the seconds per decision."""
    config_path, planner, seed, value_net = job
    env_config, reward_config = build_configs(load_config(config_path))
    env = RendezvousEnv(env_config, reward_config)
    value_fn = None
    if planner.value == "learned":
        nets = [ValueNet.load(path) for path in sorted(Path(value_net).glob("value_*.npz"))]
        value_fn = LearnedValue(env, nets)
    controller = SamplingPlanner(env, planner, seed=seed, value_fn=value_fn)
    began = time.perf_counter()
    with np.errstate(all="ignore"):   # spurious matmul warnings of Accelerate on macOS
        run = rollout(env, controller, seed)
    per_decision = (time.perf_counter() - began) / len(run.positions)
    return run, per_decision


def main() -> None:
    args = parse_args()
    seeds = range(HELD_OUT_SEED, HELD_OUT_SEED + args.episodes)
    edges = list(zip(SECTORS[:-1], SECTORS[1:], strict=True))
    results = {}
    print(f"{'planner':<22s} {'docked':>8s} {'viol.':>5s} {'Δv':>9s} {'time':>7s} "
          + " ".join(f"{f'{a:.0f}-{b:.0f}°':>8s}" for a, b in edges) + "   ms/decision")
    for setting in args.settings:
        planner = planner_config(setting, args.init_std)
        with ProcessPoolExecutor(args.workers) as pool:
            runs = list(pool.map(fly, [(args.config, planner, s, args.value_net) for s in seeds]))
        rollouts = [run for run, _ in runs]
        docked = [r for r in rollouts if r.outcome is Outcome.DOCKED]
        sectors = docked_by_sector(rollouts, SECTORS)
        violations = sum(r.outcome is Outcome.KEEP_OUT for r in rollouts)
        dv = float(np.median([r.delta_v for r in docked])) if docked else float("nan")
        t = float(np.median([r.time for r in docked])) if docked else float("nan")
        ms = 1000 * float(np.median([seconds for _, seconds in runs]))
        print(f"{setting:<22s} {len(docked):>4d}/{len(runs):<3d} {violations:>5d} "
              f"{dv:>5.2f} m/s {t:>5.0f} s "
              + " ".join(f"{d:>4d}/{n:<3d}" for d, n in sectors) + f"   {ms:>6.0f}")
        results[setting] = {
            "docked": len(docked), "episodes": len(runs), "violations": violations,
            "outcomes": {o.value: sum(r.outcome is o for r in rollouts)
                         for o in {r.outcome for r in rollouts}},
            "delta_v_median": dv, "time_median": t, "ms_per_decision": ms,
            "by_sector": [{"from_deg": a, "to_deg": b, "docked": d, "starts": n}
                          for (a, b), (d, n) in zip(edges, sectors, strict=True)],
        }
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "w") as handle:
        json.dump({"description": "Sampling planner (cross-entropy method) on the oriented "
                   "target, 200 held-out starts; value:horizon:block per setting.",
                   "results": results}, handle, indent=1)
    print(f"Results saved to {args.results}")


if __name__ == "__main__":
    main()
