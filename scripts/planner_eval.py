"""Compare ways of choosing the waypoint, with the same two frozen pilots, on the same starts.

Three planners on the 200 held-out starts of the oriented target:

- the rule of the V-bar procedure: a waypoint beside the sphere, 50 m out on
  the side of the start, only if the straight way to the hold point crosses it;
- the learned planner, if given: a policy choosing from the menu;
- the oracle, with ``--oracle``: every choice of the menu flown from every
  start and the best one kept, docking first, then least delta-v. No planner
  choosing from this menu, with these pilots, can do better, so it measures
  how much room the rule leaves.

Prints dockings, violations, median delta-v and time, dockings by start angle,
and how often the learned planner and the oracle choose the rule's side.

Usage:
    python scripts/planner_eval.py --oracle
    python scripts/planner_eval.py --planner models/planner_seed0_best.zip --oracle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from orbital_rendezvous.evaluation import HELD_OUT_SEED, docked_by_sector, evaluate
from orbital_rendezvous.rewards import Outcome
from orbital_rendezvous.utils import load_config, make_env

SECTORS = (0.0, 45.0, 90.0, 135.0, 180.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/ppo_planner.yaml")
    parser.add_argument("--go-to", default=None, help="Overrides planner.go_to.")
    parser.add_argument("--final", default=None, help="Overrides planner.final.")
    parser.add_argument("--planner", nargs="*", default=[], help="Learned planners.")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--results", default="assets/planner_evaluation.json")
    return parser.parse_args()


def summary(outcomes, delta_vs, times, violated, runs_for_sectors) -> dict:
    docked = np.array([o is Outcome.DOCKED for o in outcomes])
    return {
        "docked": int(docked.sum()),
        "attempts": len(outcomes),
        "violations": int(np.sum(violated)),
        "delta_v_median": float(np.median(np.array(delta_vs)[docked])) if docked.any() else None,
        "time_median": float(np.median(np.array(times)[docked])) if docked.any() else None,
        "docked_by_sector": docked_by_sector(runs_for_sectors, SECTORS),
    }


def row(name: str, s: dict) -> str:
    cost = (f"{s['delta_v_median']:5.2f} m/s {s['time_median']:6.0f} s"
            if s["docked"] else f"{'-':>9s} {'-':>8s}")
    sectors = "  ".join(f"{d:>3d}/{n:<3d}" for d, n in s["docked_by_sector"])
    return (f"{name:<30s} {s['docked']:>3d}/{s['attempts']:<3d} {s['violations']:>5d}  "
            f"{cost}   {sectors}")


def side(waypoint) -> int:
    """-1 or +1 for a waypoint on either side of the docking axis, 0 for none or on it."""
    return 0 if waypoint is None or abs(waypoint[0]) < 1e-6 else int(np.sign(waypoint[0]))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.go_to:
        config["planner"]["go_to"] = args.go_to
    if args.final:
        config["planner"]["final"] = args.final
    env = make_env(config)
    seeds = list(range(HELD_OUT_SEED, HELD_OUT_SEED + args.episodes))
    rule = env.pilot.planner
    results, choices = {}, {}

    runs = evaluate(env.corridor, env.pilot, seeds)
    results["rule (V-bar plan)"] = summary(
        [r.outcome for r in runs], [r.delta_v for r in runs], [r.time for r in runs],
        [r.violated for r in runs], runs)
    rule_sides = []
    for seed in seeds:
        env.corridor.reset(seed=seed)
        rule_sides.append(side(rule(env.corridor.state)))

    for path in args.planner:
        from stable_baselines3 import PPO

        model = PPO.load(path, device="cpu")
        runs = evaluate(env, lambda e, obs, m=model: m.predict(obs, deterministic=True)[0],
                        seeds)
        name = Path(path).stem
        results[name] = summary([r.outcome for r in runs], [r.delta_v for r in runs],
                                [r.time for r in runs], [r.violated for r in runs], runs)
        picks = []
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            picks.append(int(model.predict(obs, deterministic=True)[0]))
        choices[name] = picks

    if args.oracle:
        best = []
        for seed in seeds:
            flights = []
            for action in range(env.action_space.n):
                env.reset(seed=seed)
                info = env.fly(action)
                flights.append((info["outcome"] is not Outcome.DOCKED, info["delta_v"],
                                action, info, env.corridor.steps))
            best.append(min(flights, key=lambda f: (f[0], f[1])))
        starts = []
        for seed in seeds:
            env.corridor.reset(seed=seed)
            starts.append(env.corridor.state.copy())
        oracle_runs = [type("Run", (), {"start": s, "outcome": f[3]["outcome"]})
                       for s, f in zip(starts, best, strict=True)]
        results["oracle (best of the menu)"] = summary(
            [f[3]["outcome"] for f in best], [f[1] for f in best],
            [f[4] * env.config.time_step for f in best],
            [f[3]["keep_out_violated"] for f in best], oracle_runs)
        choices["oracle (best of the menu)"] = [f[2] for f in best]

    labels = "  ".join(f"{f'{a:.0f}-{b:.0f}°':>7s}"
                       for a, b in zip(SECTORS[:-1], SECTORS[1:], strict=True))
    print(f"{'':<58s}   docked, by start angle from the docking axis")
    print(f"{'planner':<30s} {'docked':>7s} {'viol.':>5s}  {'Δv':>9s} {'time':>8s}   {labels}")
    for name, s in results.items():
        print(row(name, s))
    agreement = {}
    for name, picks in choices.items():
        sides = [side(env.waypoint(k)) for k in picks]
        behind = [i for i, s in enumerate(rule_sides) if s != 0]
        same = sum(sides[i] == rule_sides[i] for i in behind)
        direct = sum(k == 0 for k in picks)
        agreement[name] = {"behind": len(behind), "same_side_as_rule": same, "direct": direct}
        print(f"{name}: straight to the hold point on {direct} starts; where the rule goes "
              f"around ({len(behind)} starts), same side as the rule on {same}")

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps({
        "description": "Choosing the waypoint with the same frozen pilots, on "
                       f"{args.episodes} held-out starts of the oriented target.",
        "pilots": {"go_to": config["planner"]["go_to"], "final": config["planner"]["final"]},
        "results": results,
        "agreement_with_rule": agreement,
    }, indent=1))
    print(f"Results saved to {args.results}")


if __name__ == "__main__":
    main()
