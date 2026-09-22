"""Evaluate a trained policy and compare it with the LQR baseline.

Reports, over a fixed set of seeded initial conditions: success rate, total
delta-v, time to dock, and the final miss distance.

Usage:
    python scripts/evaluate.py --model models/ppo_rendezvous.zip --episodes 200
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ppo_default.yaml")
    parser.add_argument("--model", default="models/ppo_rendezvous.zip")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--baseline", action="store_true", help="Also run the LQR controller.")
    return parser.parse_args()


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
