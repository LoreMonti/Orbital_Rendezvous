"""Train the planner on the oracle's ranking of every choice, instead of by trial and error.

For each training start (seeds apart from the validation and test starts),
every choice of the menu is flown by the frozen pilots and its cost
``J = fuel_weight * delta_v + time_weight * T`` recorded, in parallel. The
planner's categorical policy, the same network as the PPO planner's, is then
fitted to soft targets that favour the cheapest choices that dock, with a
term charging the expected cost of its choice, a failure priced as in the
reward (see `orbital_rendezvous.imitation`), and saved as a Stable-Baselines3 model, to be
evaluated with `planner_eval.py` like any other planner.

The labels are saved and reused: a second run with other training settings
does not fly them again.

Usage:
    python scripts/imitate_oracle.py
    python scripts/imitate_oracle.py --starts 4000 --jobs 4 --seed 1 \\
        --output models/planner_oracle_seed1.zip
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from orbital_rendezvous.hierarchy import oracle_costs
from orbital_rendezvous.utils import load_config, make_env

_ENV = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/ppo_planner_relative.yaml",
                        help="Pilots, menu and the weights of the cost J.")
    parser.add_argument("--starts", type=int, default=4000)
    parser.add_argument("--first-seed", type=int, default=50_000,
                        help="Training starts: apart from validation (90 000) and test (10 000).")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--labels", default="runs/oracle_labels.npz")
    parser.add_argument("--temperature", type=float, default=1.0, help="In units of cost J.")
    # 0 gave 197 dockings of 200, 1 gave 198, 5 gave 199, 20 and 50 gave 200.
    parser.add_argument("--risk-weight", type=float, default=20.0,
                        help="Weight of the expected cost, failures priced in; 0 for targets only.")
    parser.add_argument("--failure-cost", type=float, default=200.0,
                        help="Price of a failure, as between docking and failing in the reward.")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="models/planner_oracle.zip")
    return parser.parse_args()


def _init(config_path: str) -> None:
    global _ENV
    _ENV = make_env(load_config(config_path))


def _label(seed: int):
    return oracle_costs(_ENV, seed)


def labels(args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(args.labels)
    if path.exists():
        data = np.load(path)
        if len(data["observations"]) >= args.starts:
            print(f"Labels reused from {path}")
            n = args.starts
            return data["observations"][:n], data["costs"][:n], data["docked"][:n]
    seeds = range(args.first_seed, args.first_seed + args.starts)
    start = time.perf_counter()
    with Pool(args.jobs, initializer=_init, initargs=(args.config,)) as pool:
        rows = []
        for i, row in enumerate(pool.imap(_label, seeds, chunksize=8), 1):
            rows.append(row)
            if i % 250 == 0:
                print(f"  {i} / {args.starts} starts labelled, "
                      f"{(time.perf_counter() - start) / 60:.1f} min", flush=True)
    observations, costs, docked = (np.array(x) for x in zip(*rows, strict=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, observations=observations, costs=costs, docked=docked)
    print(f"Labels saved to {path}")
    return observations, costs, docked


def main() -> None:
    args = parse_args()
    from stable_baselines3 import PPO

    from orbital_rendezvous.imitation import expected_costs, imitate, soft_targets

    config = load_config(args.config)
    observations, costs, docked = labels(args)
    usable = docked.any(axis=1)
    print(f"{usable.sum()} of {len(usable)} starts have a choice that docks, and are used")
    targets = soft_targets(costs[usable], docked[usable], args.temperature)

    env = make_env(config)
    model = PPO(config["training"]["policy"], env, seed=args.seed,
                policy_kwargs=config["training"].get("policy_kwargs"), device="cpu")
    penalties = expected_costs(costs[usable], docked[usable], args.failure_cost)
    history = imitate(model, observations[usable], targets, penalties, args.risk_weight,
                      epochs=args.epochs, seed=args.seed)
    best = min(history, key=lambda h: h["validation_loss"])
    print(f"Best validation loss {best['validation_loss']:.3f} at epoch {best['epoch']}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    output.with_suffix(".json").write_text(json.dumps(
        {"temperature": args.temperature, "risk_weight": args.risk_weight,
         "failure_cost": args.failure_cost, "starts": int(usable.sum()), "history": history},
        indent=1))
    print(f"Planner saved to {output}")


if __name__ == "__main__":
    main()
