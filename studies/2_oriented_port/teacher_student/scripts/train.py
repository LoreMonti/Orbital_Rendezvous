"""Train a pilot or a planner of the study with PPO, headless.

The package's `studies/1_free_docking/scripts/train.py` builds the rendezvous environment; the
study's configurations need `teacher_student.config.make_env`, which also
builds the go-to task (a ``goal`` section) and the planner's environment (a
``planner`` section). Everything else is the package's training.

Usage:
    python studies/2_oriented_port/teacher_student/scripts/train.py \
        --config studies/2_oriented_port/teacher_student/configs/ppo_goto.yaml \
        --output models/goto_seed0.zip
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# The study's package, next to this folder: studies/2_oriented_port/teacher_student/teacher_student.
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime
from pathlib import Path

from orbital_rendezvous.core.utils import load_config
from orbital_rendezvous.rl.training import train
from teacher_student.config import make_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    seed = config["training"]["seed"] if args.seed is None else args.seed
    steps = int(args.timesteps or config["training"]["total_timesteps"])
    run_dir = Path(args.runs) / datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Training for {steps:,} steps; run directory {run_dir}")
    train(config, seed, steps, run_dir, Path(args.output), env_factory=make_env)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
