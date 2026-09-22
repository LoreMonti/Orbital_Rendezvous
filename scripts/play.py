"""Watch the trained agent dock, or fly the chaser yourself and compare.

Usage:
    python scripts/play.py --model models/ppo_rendezvous.zip
    python scripts/play.py --human

In human mode the arrow keys fire the thrusters, and the run ends with your
total delta-v next to the agent's and the LQR's on the same initial condition.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ppo_default.yaml")
    parser.add_argument("--model", default="models/ppo_rendezvous.zip")
    parser.add_argument("--human", action="store_true", help="Fly the chaser yourself.")
    parser.add_argument("--record", default=None, help="Save the run as a GIF at this path.")
    return parser.parse_args()


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
