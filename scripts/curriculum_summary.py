"""The two curricula of Step 15 during training, next to the cone curriculum of Step 14.

Phase 1 narrows the approach cone for starts in front of the port only; phase
2 keeps the cone at 15 degrees and widens the starts towards the back of the
station. Draws one panel per phase: in the first, the cone of Step 14, with
starts in every direction, for comparison; in the second, the widest start
angle mastered, with the line at 90 degrees beyond which the chaser must go
around the station.

`--collect` first gathers the `curriculum.json` of each training run, labelled
by the seed in its `config.yaml`, into `assets/start_curriculum.json`.

Usage:
    python scripts/curriculum_summary.py --collect runs/<run 1> runs/<run 2> runs/<run 3>
    python scripts/curriculum_summary.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orbital_rendezvous.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--collect", nargs="*", default=None, help="Run directories to gather.")
    parser.add_argument("--results", default="assets/start_curriculum.json")
    parser.add_argument("--step14", default="assets/cone_curriculum.json")
    parser.add_argument("--plot", default="assets/start_curriculum.png")
    return parser.parse_args()


def collect(run_dirs: list[str], results: Path) -> None:
    runs = {}
    for run_dir in map(Path, run_dirs):
        seed = load_config(run_dir / "config.yaml")["training"]["seed"]
        runs[f"seed{seed}"] = json.loads((run_dir / "curriculum.json").read_text())
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps({
        "description": "Step 15: phase 1 narrows the approach cone for starts at most 15 "
                       "degrees from the docking axis; phase 2 keeps a 15 degree cone and "
                       "widens the starts. Measured every 10 rollouts on 20 deterministic "
                       "attempts under the strict rule.",
        "runs": dict(sorted(runs.items())),
    }, indent=1))
    print(f"{len(runs)} runs gathered into {results}")


def main() -> None:
    args = parse_args()
    if args.collect:
        collect(args.collect, Path(args.results))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from orbital_rendezvous.game_view import AMBER, BLUE, GREEN, GRID, MUTED, PANEL, RED, TEXT

    runs = json.loads(Path(args.results).read_text())["runs"]
    step14 = json.loads(Path(args.step14).read_text())["runs"]
    colours = (AMBER, GREEN, BLUE)

    fig, (cone_ax, start_ax) = plt.subplots(
        2, 1, figsize=(9.5, 8.0), facecolor=PANEL,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.38},
    )
    for ax in (cone_ax, start_ax):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(alpha=0.25, color=GRID)
        ax.set_xlabel("training steps [millions]", color=TEXT)
        ax.set_ylim(0, 195)
        ax.set_yticks([15, 45, 90, 135, 180])

    def curve(ax, history, key, colour, start, **style):
        # History rows hold the stage after each measurement: prepend where it began.
        steps = [start[0]] + [h["timesteps"] / 1e6 for h in history]
        values = [start[1]] + [h[key] for h in history]
        ax.step(steps, values, where="post", color=colour, **style)
        return steps[-1], values[-1]

    # Phase 1, with Step 14 faded behind it.
    end14 = 0.0
    for history in step14.values():
        x, y = curve(cone_ax, history, "cone_deg", MUTED, (0.0, 180.0), lw=1.4, alpha=0.6)
        end14 = max(end14, x)
    cone_ax.text(end14, 90 - 6, "Step 14, starts in every direction: stops at 90°",
                 ha="right", va="top", color=MUTED, fontsize=9)
    ends = []
    for (name, history), colour in zip(runs.items(), colours, strict=False):
        if history["cone"]:
            x, y = curve(cone_ax, history["cone"], "cone_deg", colour, (0.0, 180.0), lw=2.2)
            ends.append((name, colour, x, y))
    # The labels in a column right of where the curves end, clear of all of them.
    column = max((x for _, _, x, _ in ends), default=0.0) + 0.3
    for i, (name, colour, x, y) in enumerate(ends):
        cone_ax.text(column, 60 - 13 * i, f"{name.replace('seed', 'seed ')}: {y:.0f}° at {x:.1f} M",
                     color=colour, fontsize=9, va="center")
    cone_ax.axhline(15, color=TEXT, ls=":", lw=1.0)
    cone_ax.set_xlim(0, end14 * 1.02)
    cone_ax.set_ylabel("cone half-angle [°]", color=TEXT)
    cone_ax.set_title("Phase 1 — starts in front of the port: the cone narrows to 15°",
                      color=TEXT, fontsize=11, loc="left")

    # Phase 2.
    end = 0.0
    for (name, history), colour in zip(runs.items(), colours, strict=False):
        if history["starts"]:
            # Phase 2 begins where phase 1 ends, at the first stage of the starts.
            begin = history["cone"][-1]["timesteps"] / 1e6 if history["cone"] else 0.0
            x, y = curve(start_ax, history["starts"], "start_deg", colour, (begin, 15.0),
                         lw=2.2)
            end = max(end, x)
            start_ax.annotate(f"{name.replace('seed', 'seed ')}: {y:.0f}°", (x, y),
                              xytext=(6, 0), textcoords="offset points", color=colour,
                              fontsize=9, va="center")
    start_ax.axhline(90, color=RED, ls="--", lw=1.2)
    start_ax.text(0.99, 92, "beyond 90°: the chaser must go around the station",
                  transform=start_ax.get_yaxis_transform(), color=RED, fontsize=9, va="bottom",
                  ha="right")
    start_ax.axhline(180, color=TEXT, ls=":", lw=1.0)
    start_ax.text(0.01, 182, "180°: starts directly behind it",
                  transform=start_ax.get_yaxis_transform(), color=TEXT, fontsize=9, va="bottom")
    start_ax.set_xlim(0, end * 1.12)
    start_ax.set_ylabel("widest start angle reached [°]", color=TEXT)
    start_ax.set_title("Phase 2 — cone at 15°: the starts widen towards the back",
                       color=TEXT, fontsize=11, loc="left")

    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=130, bbox_inches="tight", facecolor=PANEL)
    plt.close(fig)
    print(f"Plot saved to {args.plot}")


if __name__ == "__main__":
    main()
