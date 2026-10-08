"""Go-Explore phase 1 on many starts of the oriented target, in parallel (Step 27).

Explores each start with `go_explore.explore`, replays the best docking found
from a fresh reset to check it, and appends one line per start to
``results.jsonl`` in the run directory, with the thrusts of the docking in
``trajectories/<seed>.npy``. The starts are drawn from the task itself, every
direction from 80 to 200 m, with seeds from ``--first-seed`` on, far from the
held-out seeds of the evaluation (10 000 to 10 199), so that phase 2 can be
tested on starts it never saw. Resumable: starts already in ``results.jsonl``
are skipped.

Usage:
    python studies/2_oriented_port/scripts/go_explore.py --run runs/go_explore --starts 400
    python studies/2_oriented_port/scripts/go_explore.py --run runs/go_explore --workers 4
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from orbital_rendezvous import RendezvousEnv
from orbital_rendezvous.core.evaluation import start_angle
from orbital_rendezvous.core.utils import build_configs, load_config
from orbital_rendezvous.planning.go_explore import GoExploreConfig, explore, replay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="studies/2_oriented_port/configs/ppo_corridor.yaml")
    parser.add_argument("--run", default="runs/go_explore")
    parser.add_argument("--starts", type=int, default=400)
    parser.add_argument("--first-seed", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--budget", type=int, default=GoExploreConfig.budget)
    parser.add_argument("--after-docking", type=int, default=GoExploreConfig.after_docking)
    return parser.parse_args()


def run_one(job: tuple[str, int, GoExploreConfig, str]) -> dict:
    config_path, seed, settings, out = job
    env_config, reward_config = build_configs(load_config(config_path))
    env = RendezvousEnv(env_config, reward_config)
    began = time.perf_counter()
    with np.errstate(all="ignore"):   # spurious matmul warnings of Accelerate on macOS
        found = explore(env, seed, settings)
        check = replay(env, seed, found.actions) if found.docked else None
    if found.docked:
        np.save(Path(out) / "trajectories" / f"{seed}.npy", found.actions)
    return {
        "seed": seed,
        "start_r": round(float(np.hypot(*found.start[:2])), 1),
        "start_deg": round(start_angle(found.start), 1),
        "docked": found.docked,
        "replay": None if check is None else check["outcome"].value,
        "violated": None if check is None else check["violated"],
        "delta_v": None if check is None else round(check["delta_v"], 3),
        "time_s": None if check is None else check["time"],
        "first_docking": found.first_docking,
        "rounds": found.rounds,
        "cells": found.cells,
        "closest_m": round(found.closest, 1),
        "seconds": round(time.perf_counter() - began, 1),
    }


def main() -> None:
    args = parse_args()
    run = Path(args.run)
    (run / "trajectories").mkdir(parents=True, exist_ok=True)
    results = run / "results.jsonl"
    done = set()
    if results.exists():
        done = {json.loads(line)["seed"] for line in results.read_text().splitlines() if line}
    settings = GoExploreConfig(budget=args.budget, after_docking=args.after_docking)
    seeds = [s for s in range(args.first_seed, args.first_seed + args.starts) if s not in done]
    print(f"{len(done)} starts done, {len(seeds)} to go", flush=True)
    with ProcessPoolExecutor(args.workers) as pool, open(results, "a") as handle:
        jobs = [(args.config, s, settings, str(run)) for s in seeds]
        for row in pool.map(run_one, jobs):
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(row, flush=True)
    rows = [json.loads(line) for line in results.read_text().splitlines() if line]
    docked = [r for r in rows if r["replay"] == "docked"]
    print(f"Docked on replay: {len(docked)} of {len(rows)}")


if __name__ == "__main__":
    main()
