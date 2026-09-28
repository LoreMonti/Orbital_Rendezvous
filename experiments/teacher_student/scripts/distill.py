"""Distil the planner and the two pilots into one network, the student.

The teacher, the system of Step 17c, flies training starts (seeds apart from
validation and test), a share of them with noise added to the thrust it
applies but not to the labels it gives; the flights are saved and reused. The
student, a single network with two heads (see `teacher_student.distillation`),
is then fitted to them, and every few epochs flown on 50 validation starts;
the student that docks most often, cheaper on a tie, is saved. Evaluate it
with ``corridor_eval.py --student``.

Usage:
    python scripts/distill.py
    python scripts/distill.py --seed 1 --output models/student_seed1.pt
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# The study's package, next to this folder: experiments/teacher_student/teacher_student.
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# Default paths are relative to the repository root, where the scripts are run.
STUDY = "experiments/teacher_student"

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from orbital_rendezvous.evaluation import evaluate
from orbital_rendezvous.rewards import Outcome
from orbital_rendezvous.utils import load_config
from teacher_student.config import make_env
from teacher_student.distillation import load_flights, save_flights

_ENV = None
_PILOT = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=f"{STUDY}/configs/ppo_planner_relative.yaml",
                        help="The teacher's pilots and menu, and the weights of the cost J.")
    parser.add_argument("--planner", default="models/planner_oracle_seed0.zip",
                        help="The teacher's planner, from Step 17c.")
    parser.add_argument("--episodes", type=int, default=4000, help="Per noise level.")
    parser.add_argument("--noise", nargs="*", type=float, default=[0.0, 0.02, 0.1],
                        help="Standard deviations of the noise on the applied thrust, per level.")
    parser.add_argument("--noise-radius", nargs="*", type=float, default=[0.0, 0.0, 40.0],
                        help="For each level, the distance from the station within which no "
                             "noise is added.")
    parser.add_argument("--first-seed", type=int, default=50_000)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--data", default="runs/teacher_flights.npz")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--near-weight", type=float, default=10.0,
                        help="Weight of the steps within 20 m of the station in the thrust loss.")
    parser.add_argument("--final-learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=1.0, help="Weight of the mode loss.")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--validation", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="models/student_seed0.pt")
    return parser.parse_args()


def _init(config_path: str, planner_path: str) -> None:
    global _ENV, _PILOT
    from stable_baselines3 import PPO

    from teacher_student.distillation import learned_planner

    _ENV = make_env(load_config(config_path))
    _PILOT = _ENV.pilot
    _PILOT.planner = learned_planner(PPO.load(planner_path, device="cpu"), _ENV)


def _fly(task: tuple[int, float, float]) -> dict:
    from teacher_student.distillation import teacher_flight

    seed, noise, radius = task
    rng = np.random.default_rng([seed, int(1000 * noise)])
    return dict(teacher_flight(_ENV, _PILOT, seed, noise, rng, radius), noise=noise)


def flights(args) -> list[dict]:
    path = Path(args.data)
    if path.exists():
        print(f"Teacher flights reused from {path}")
        return load_flights(path)
    seeds = range(args.first_seed, args.first_seed + args.episodes)
    levels = list(zip(args.noise, args.noise_radius, strict=True))
    tasks = [(s, noise, radius) for noise, radius in levels for s in seeds]
    start = time.perf_counter()
    with Pool(args.jobs, initializer=_init, initargs=(args.config, args.planner)) as pool:
        rows = pool.map(_fly, tasks, chunksize=16)
    print(f"{len(rows)} teacher flights in {(time.perf_counter() - start) / 60:.1f} min")
    save_flights(rows, path)
    print(f"Teacher flights saved to {path}")
    return rows


def main() -> None:
    args = parse_args()
    from teacher_student.distillation import (
        MODES,
        Student,
        StudentPilot,
        save_student,
        train_student,
    )

    rows = flights(args)
    for noise in sorted({r["noise"] for r in rows}):
        part = [r for r in rows if r["noise"] == noise]
        print(f"noise {noise}: teacher docked {sum(r['docked'] for r in part)} / {len(part)}")
    modes = np.bincount([r["mode"] for r in rows], minlength=len(MODES))
    print("teacher's modes: " + ", ".join(f"{m} {n}" for m, n in zip(MODES, modes, strict=True)))

    planner = make_env(load_config(args.config))
    corridor = planner.corridor
    seeds = range(90_000, 90_000 + args.validation)

    def closed_loop(student) -> tuple[float, float]:
        runs = evaluate(corridor, StudentPilot(student), seeds)
        docked = [planner.fuel_weight * r.delta_v + planner.time_weight * r.time
                  for r in runs if r.outcome is Outcome.DOCKED]
        return len(docked) / len(runs), float(np.median(docked)) if docked else float("inf")

    student = Student(hidden=args.hidden)
    history = train_student(student, rows, epochs=args.epochs, beta=args.beta,
                            evaluate=closed_loop, evaluate_all_from=args.epochs // 2,
                            near_weight=args.near_weight,
                            position_scale=corridor.config.max_distance,
                            final_learning_rate=args.final_learning_rate, seed=args.seed)
    for row in history:
        if "success" in row:
            print(f"  epoch {row['epoch'] + 1:3d}: thrust loss {row['thrust_loss']:.4f}, "
                  f"mode loss {row['mode_loss']:.3f}, "
                  f"validation docked {100 * row['success']:.0f} %, J {row['cost']:.1f}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_student(student, output)
    output.with_suffix(".json").write_text(json.dumps(
        {"planner": args.planner, "noise": args.noise, "noise_radius": args.noise_radius,
         "beta": args.beta, "near_weight": args.near_weight,
         "final_learning_rate": args.final_learning_rate, "history": history},
        indent=1))
    print(f"Student saved to {output}")


if __name__ == "__main__":
    main()
