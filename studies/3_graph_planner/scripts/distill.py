"""Distil the graph pilot into one network (Step 28, block 5).

Three phases, each skipped if its output exists:

1. **collect**: the graph pilot flies ``--flights`` training starts, in
   parallel, with DART noise on the thrust applied; the student's observations,
   the teacher's commands and the mode of its way are saved to
   ``<run>/flights.npz``.
2. **train**: the two-headed student is fitted to them, the student that docks
   most often on 100 validation starts kept, and saved to ``<run>/student.pt``.
3. **evaluate**: the student flies the 200 held-out starts of Parts 2 and 3,
   with and without thrust errors; a row is added to
   ``studies/3_graph_planner/assets/student.json``.

Training, validation and held-out starts use disjoint seeds.

Usage:
    python studies/3_graph_planner/scripts/distill.py --run runs/distill/seed0
    python studies/3_graph_planner/scripts/distill.py --run runs/distill/seed1 --seed 1
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
from orbital_rendezvous.planning.distill import (
    Student,
    StudentPilot,
    load_student,
    save_student,
    teacher_flight,
    train_student,
)
from orbital_rendezvous.planning.graph import CWGraph, GraphPilot

TRAIN_SEED, VALIDATION_SEED = 200_000, 300_000
SECTORS = (0.0, 45.0, 90.0, 135.0, 180.0)
EDGES = list(zip(SECTORS[:-1], SECTORS[1:], strict=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="studies/2_oriented_port/configs/ppo_corridor.yaml")
    parser.add_argument("--run", default="runs/distill/seed0")
    parser.add_argument("--seed", type=int, default=0, help="of the student's training")
    parser.add_argument("--flights", type=int, default=12_000)
    parser.add_argument("--noise", type=float, default=0.1, help="beyond 40 m")
    parser.add_argument("--noise-near", type=float, default=0.02, help="within 40 m")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--results", default="studies/3_graph_planner/assets/student.json")
    return parser.parse_args()


def make_env(config_path: str) -> RendezvousEnv:
    return RendezvousEnv(*build_configs(load_config(config_path)))


def collect_one(job):
    config_path, seed, noise, noise_near = job
    env = make_env(config_path)
    pilot = GraphPilot(env, CWGraph(make_env(config_path)), seed=seed)
    with np.errstate(all="ignore"):   # spurious matmul warnings of Accelerate on macOS
        return teacher_flight(env, pilot, seed, noise, np.random.default_rng(seed),
                              noise_near=noise_near)


def fly_student(job):
    config_path, student_path, seed, magnitude, angle = job
    from orbital_rendezvous.core.evaluation import rollout as fly

    env = make_env(config_path)
    pilot = StudentPilot(load_student(student_path))
    rng = np.random.default_rng(seed)

    def controller(e, obs):
        action = pilot(e, obs)
        if not (magnitude or angle):
            return action
        scale, turn = 1.0 + magnitude * rng.normal(), np.radians(angle) * rng.normal()
        c, s = np.cos(turn), np.sin(turn)
        return np.clip(scale * np.array([c * action[0] - s * action[1],
                                         s * action[0] + c * action[1]]), -1.0, 1.0)

    with np.errstate(all="ignore"):
        return fly(env, controller, seed)


def main() -> None:
    args = parse_args()
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)

    flights_path = run / "flights.npz"
    if not flights_path.exists():
        began = time.time()
        jobs = [(args.config, TRAIN_SEED + i, args.noise, args.noise_near)
                for i in range(args.flights)]
        with ProcessPoolExecutor(args.workers) as pool:
            flights = list(pool.map(collect_one, jobs, chunksize=16))
        lengths = np.array([len(f["actions"]) for f in flights])
        np.savez(flights_path,
                 observations=np.concatenate([f["observations"] for f in flights]),
                 actions=np.concatenate([f["actions"] for f in flights]),
                 lengths=lengths, modes=np.array([f["mode"] for f in flights]),
                 docked=np.array([f["docked"] for f in flights]))
        print(f"collected {len(flights)} flights, {sum(f['docked'] for f in flights)} docked "
              f"under noise, {lengths.sum()} steps, {(time.time() - began) / 60:.1f} min")
    with np.load(flights_path) as data:   # read every array once
        observations, actions = data["observations"], data["actions"]
        lengths, modes = data["lengths"], data["modes"]
    ends = np.cumsum(lengths)
    flights = [{"observations": observations[e - n:e], "actions": actions[e - n:e], "mode": m}
               for e, n, m in zip(ends, lengths, modes, strict=True)]

    student_path = run / "student.pt"
    if not student_path.exists():
        validation = [(args.config, VALIDATION_SEED + i) for i in range(100)]

        def evaluate(student: Student) -> tuple[float, float]:
            pilot, env = StudentPilot(student), make_env(args.config)
            runs = [rollout(env, pilot, seed) for _, seed in validation]
            docked = [r for r in runs if r.outcome is Outcome.DOCKED]
            cost = float(np.median([r.delta_v for r in docked])) if docked else float("inf")
            print(f"  validation: {len(docked)} of {len(runs)} docked", flush=True)
            return len(docked) / len(runs), cost

        student = Student()
        with np.errstate(all="ignore"):
            history = train_student(student, flights, epochs=args.epochs, evaluate=evaluate,
                                    seed=args.seed)
        save_student(student, student_path)
        (run / "history.json").write_text(json.dumps(history, indent=1))

    table_path = Path(args.results)
    table = json.loads(table_path.read_text()) if table_path.exists() else {"results": {}}
    for magnitude, angle in ((0.0, 0.0), (0.1, 3.0)):
        jobs = [(args.config, str(student_path), HELD_OUT_SEED + i, magnitude, angle)
                for i in range(200)]
        with ProcessPoolExecutor(args.workers) as pool:
            runs = list(pool.map(fly_student, jobs))
        docked = [r for r in runs if r.outcome is Outcome.DOCKED]
        outcomes = {o.value: sum(r.outcome is o for r in runs) for o in {r.outcome for r in runs}}
        sectors = docked_by_sector(runs, SECTORS)
        dv = float(np.median([r.delta_v for r in docked])) if docked else float("nan")
        t = float(np.median([r.time for r in docked])) if docked else float("nan")
        name = f"student {run.name}" + (f", thrust error {magnitude:.2f}, {angle:.0f} deg"
                                        if magnitude or angle else "")
        print(f"{name}: {len(docked)}/200 docked, {outcomes}, {dv:.2f} m/s, {t:.0f} s")
        print("by sector:", sectors)
        table["results"][name] = {
            "docked": len(docked), "episodes": 200, "outcomes": outcomes,
            "delta_v_median": dv, "time_median": t, "flights": len(flights),
            "by_sector": [{"from_deg": a, "to_deg": b, "docked": d, "starts": n}
                          for (a, b), (d, n) in zip(EDGES, sectors, strict=True)],
        }
    table["description"] = ("The graph pilot distilled into one network with DART "
                            "(scripts/distill.py), on the 200 held-out starts.")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(json.dumps(table, indent=1))
    print(f"Results saved to {table_path}")


if __name__ == "__main__":
    main()
