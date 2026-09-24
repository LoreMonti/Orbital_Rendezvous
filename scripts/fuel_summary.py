"""The whole fuel story in one plot: the default agent and the best of each attempt.

Reads the results already saved by `evaluate.py` and `fuel_study.py` (no
training), picks from each study the cheapest configuration that docks
reliably on every seed, and draws them in order, joined by arrows, against the
LQR front and the two-impulse reference.

Usage:
    python scripts/fuel_summary.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Each attempt, in the order it was made: its label and the file of its study.
ATTEMPTS = (
    ("fuel weight and curriculum\n(Step 11)", "assets/fuel_study.json"),
    ("Lagrange multiplier\n(Step 12)", "assets/lagrange_study.json"),
    ("engine switch and Lagrange\n(Step 13)", "assets/lagrange_study_switch.json"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--baselines", default="assets/evaluation.json")
    parser.add_argument("--plot", default="assets/fuel_journey.png")
    return parser.parse_args()


def best_reliable(path: str) -> dict | None:
    """The cheapest configuration of a study that docks reliably on every seed."""
    if not Path(path).exists():
        return None
    rows = json.loads(Path(path).read_text())["configurations"]
    complete = [r for r in rows if r["seeds"] and r["reliable_seeds"] == r["seeds"]]
    return min(complete, key=lambda r: r["delta_v_median"]) if complete else None


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from orbital_rendezvous.game_view import AMBER, BLUE, GRID, MUTED, PANEL, TEXT

    baselines = json.loads(Path(args.baselines).read_text())
    agent = baselines["agent"]
    points = [("default agent", agent["time_median"], agent["delta_v_median"])]
    for label, path in ATTEMPTS:
        row = best_reliable(path)
        if row is not None:
            points.append((label, row["time_median"], row["delta_v_median"]))

    fig, ax = plt.subplots(figsize=(9.5, 5.8), facecolor=PANEL)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(alpha=0.25, color=GRID)

    front = baselines["lqr_front"]
    ax.plot([s["time_median"] for s in front], [s["delta_v_median"] for s in front],
            color=BLUE, lw=2.2)
    at = front[len(front) // 2]
    ax.annotate("LQR, best trade-offs", (at["time_median"], at["delta_v_median"]),
                xytext=(0, 10), textcoords="offset points", color=BLUE, fontsize=10)
    bound = baselines["two_impulse"]["delta_v_median"]
    ax.axhline(bound, color=TEXT, ls="--", lw=1)
    ax.text(0.98, bound, "ideal two-impulse transfer (not flyable)  ", color=TEXT,
            fontsize=9, ha="right", va="bottom", transform=ax.get_yaxis_transform())

    for (_, *start), (_, *stop) in zip(points, points[1:], strict=False):
        ax.annotate("", xy=stop, xytext=start,
                    arrowprops={"arrowstyle": "-|>", "color": AMBER, "lw": 1.6,
                                "shrinkA": 9, "shrinkB": 9})
    first = points[0][2]
    star = points[0]
    ax.scatter(star[1], star[2], s=320, marker="*", color=TEXT, edgecolor=TEXT, zorder=5)
    ax.annotate(f"{star[0]}, {star[2]:.2f} m/s", (star[1], star[2]), xytext=(-14, -2),
                textcoords="offset points", ha="right", va="center", color=TEXT, fontsize=9)
    # The attempts can land almost on top of each other, so their labels go in a
    # column in the empty lower left, each tied to its point by a thin line.
    for i, (label, time, delta_v) in enumerate(points[1:]):
        ax.scatter(time, delta_v, s=110, color=AMBER, edgecolor=TEXT, zorder=5)
        change = 100 * (delta_v / first - 1)
        ax.annotate(
            f"{label.replace(chr(10), ' ')}\n{delta_v:.2f} m/s, {change:+.0f} % fuel, "
            f"{time:.0f} s",
            (time, delta_v), xytext=(0.03, 0.62 - 0.14 * i), textcoords="axes fraction",
            ha="left", va="center", color=AMBER, fontsize=9,
            arrowprops={"arrowstyle": "-", "color": AMBER, "lw": 0.6, "alpha": 0.7},
        )

    ax.set_xlim(0, max(s["time_median"] for s in front) * 1.05)
    ax.set_ylim(0, max(p[2] for p in points + [("", 0, front[0]["delta_v_median"])]) * 1.12)
    ax.set_xlabel("time to dock, median [s]", color=TEXT)
    ax.set_ylabel(r"fuel spent, median $\Delta v$ [m/s]", color=TEXT)
    ax.set_title("Asking the agent to save fuel, attempt by attempt  (lower left is better)",
                 color=TEXT, fontsize=11, loc="left")
    ax.text(0.99, 0.03, "each point: the cheapest setting that docks on every seed",
            transform=ax.transAxes, color=MUTED, fontsize=8, ha="right")
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=130, bbox_inches="tight", facecolor=PANEL)
    plt.close(fig)
    for label, time, delta_v in points:
        print(f"{label.replace(chr(10), ' '):45s} {delta_v:.2f} m/s  {time:5.0f} s")
    print(f"Plot saved to {args.plot}")


if __name__ == "__main__":
    main()
