"""Train a PPO agent on the rendezvous task.

Usage:
    python scripts/train.py --config configs/ppo_default.yaml
    python scripts/train.py --config configs/ppo_default.yaml --no-render
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ppo_default.yaml")
    parser.add_argument("--output", default="models/ppo_rendezvous.zip")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Train without the live window, at full speed.",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
