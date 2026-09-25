"""The approach cone during training: where the curriculum stops.

Reads `assets/cone_curriculum.json` (the cone's half-angle and the docking rate
under the strict rule, measured during two training runs) and draws the cone
narrowing from 180 degrees, the line at 90 degrees where it stopped, and the
15 degrees it was meant to reach.

Usage:
    python scripts/cone_summary.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", default="assets/cone_curriculum.json")
    parser.add_argument("--plot", default="assets/cone_curriculum.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from orbital_rendezvous.game_view import AMBER, GREEN, GRID, MUTED, PANEL, RED, TEXT

    runs = json.loads(Path(args.results).read_text())["runs"]
    fig, (cone_ax, dock_ax) = plt.subplots(
        2, 1, figsize=(9.5, 6.4), facecolor=PANEL, sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.12},
    )
    for ax in (cone_ax, dock_ax):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(alpha=0.25, color=GRID)

    colours = (AMBER, GREEN)
    for (name, history), colour in zip(runs.items(), colours, strict=False):
        steps = [h["timesteps"] / 1e6 for h in history]
        cone_ax.step(steps, [h["cone_deg"] for h in history], where="post", color=colour,
                     lw=2.2, label=name.replace("seed", "training seed "))
        dock_ax.plot(steps, [100 * h["success"] for h in history], color=colour, lw=1.2)

    cone_ax.axhline(90, color=RED, ls="--", lw=1.2)
    cone_ax.text(0.99, 92, "90°: every start behind the station must now go around it",
                 transform=cone_ax.get_yaxis_transform(), ha="right", va="bottom",
                 color=RED, fontsize=9)
    cone_ax.axhline(15, color=TEXT, ls=":", lw=1.2)
    cone_ax.text(0.99, 17, "15°: the cone the curriculum was meant to reach",
                 transform=cone_ax.get_yaxis_transform(), ha="right", va="bottom",
                 color=TEXT, fontsize=9)
    cone_ax.set_ylim(0, 190)
    cone_ax.set_yticks([15, 45, 90, 135, 180])
    cone_ax.set_ylabel("cone half-angle [°]", color=TEXT)
    cone_ax.set_title("Narrowing the approach cone: both runs stop at 90°",
                      color=TEXT, fontsize=11, loc="left")
    cone_ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9,
                   loc="upper right")

    dock_ax.axhline(90, color=MUTED, ls=":", lw=1)
    dock_ax.text(0.01, 92, "needed to narrow the cone", transform=dock_ax.get_yaxis_transform(),
                 color=MUTED, fontsize=8, va="bottom")
    dock_ax.set_ylim(-5, 110)
    dock_ax.set_ylabel("docks [%]", color=TEXT)
    dock_ax.set_xlabel("training steps [millions]", color=TEXT)

    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=130, bbox_inches="tight", facecolor=PANEL)
    plt.close(fig)
    print(f"Plot saved to {args.plot}")


if __name__ == "__main__":
    main()
