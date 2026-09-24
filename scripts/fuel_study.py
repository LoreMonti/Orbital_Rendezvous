"""Trade time for fuel: train the agent over a grid of discounts and fuel weights.

With a discount gamma, a docking bonus earned after K steps is worth 100 gamma^K,
so at gamma = 0.99 docking late costs far more bonus than the fuel it saves.
This study trains every (gamma, final fuel weight) pair on several seeds,
headless and in parallel, each with a curriculum that starts the fuel weight
at 2 and raises it to its target over the first half of training. It evaluates
each model on the 200 held-out starts of `evaluate.py`, and plots every run on
the same delta-v against time axes as the LQR front and the two-impulse
reference. The default agent is untouched.

Each run writes its result to disk as soon as it ends, so an interrupted study
picks up where it stopped when started again.

Usage:
    python scripts/fuel_study.py                       # the full grid, 24 runs
    python scripts/fuel_study.py --jobs 4              # fewer runs at once
    python scripts/fuel_study.py --gammas 0.99 --fuel-weights 2 --seeds 0 \\
        --timesteps 20000 --episodes 5                  # a quick check of the pipeline
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from functools import partial
from pathlib import Path

from orbital_rendezvous.study import (
    RELIABLE,
    Job,
    aggregate,
    ignore_interrupts,
    make_jobs,
    run_job,
)
from orbital_rendezvous.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/ppo_default.yaml")
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.99, 0.999])
    parser.add_argument("--fuel-weights", type=float, nargs="+", default=[2.0, 5.0, 10.0, 20.0],
                        help="Final fuel weights, reached at the end of the curriculum.")
    parser.add_argument("--curriculum-start", type=float, default=2.0)
    parser.add_argument("--curriculum-ramp", type=float, default=0.5,
                        help="Fraction of the training over which the fuel weight rises.")
    parser.add_argument("--deadzone", type=float, default=0.0,
                        help="Minimum thruster level, as a fraction of the maximum thrust. "
                             "Off by default: it lets the agent coast, but costs the "
                             "precision of the final approach (see ROADMAP, Step 11).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--timesteps", type=int, default=4_000_000)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=6, help="Runs trained at the same time.")
    parser.add_argument("--directory", default="runs/fuel_study")
    parser.add_argument("--baselines", default="assets/evaluation.json",
                        help="LQR front and two-impulse reference, from evaluate.py.")
    parser.add_argument(
        "--budgets", type=float, nargs="+", default=None,
        help="Fuel budgets in m/s: the fuel weight becomes a Lagrange multiplier that keeps "
             "the agent within each budget, instead of the grid of fixed weights.",
    )
    parser.add_argument("--results", default=None,
                        help="Default: assets/fuel_study.json, or lagrange_study.json.")
    parser.add_argument("--plot", default=None,
                        help="Default: assets/fuel_study.png, or lagrange_study.png.")
    args = parser.parse_args()
    name = "lagrange_study" if args.budgets else "fuel_study"
    args.results = args.results or f"assets/{name}.json"
    args.plot = args.plot or f"assets/{name}.png"
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    jobs = make_jobs(args.gammas, args.fuel_weights, args.seeds, args.timesteps,
                     args.episodes, args.directory, args.curriculum_start, args.curriculum_ramp,
                     args.deadzone, args.budgets)
    done = [json.loads(job.result_path.read_text()) for job in jobs if job.result_path.exists()]
    todo = [job for job in jobs if not job.result_path.exists()]
    print(f"{len(jobs)} runs: {len(done)} already done, {len(todo)} to train, "
          f"{args.jobs} at a time, {args.timesteps:,} steps each")

    start = time.perf_counter()
    # Workers ignore Ctrl-C; the parent catches it and terminates them all. The
    # results are awaited with a short timeout so that Ctrl-C is always seen.
    pool = multiprocessing.get_context("spawn").Pool(args.jobs, initializer=ignore_interrupts)
    try:
        results = pool.imap_unordered(partial(run_job, base_config=config), todo)
        for _ in todo:
            while True:
                try:
                    result = results.next(timeout=1.0)
                    break
                except multiprocessing.TimeoutError:
                    continue
            done.append(result)
            s = result["summary"]
            minutes = (time.perf_counter() - start) / 60
            label = Job(**{k: result[k] for k in Job.__dataclass_fields__}).label
            print(f"[{len(done):2d}/{len(jobs)}, {minutes:5.1f} min] {label}: "
                  f"docked {100 * s['success_rate']:5.1f} %, "
                  f"delta-v {s['delta_v_median']:.2f} m/s, time {s['time_median']:.0f} s",
                  flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted: stopping the runs in progress. Finished runs are kept; "
              "run the same command again to resume.", flush=True)
        pool.terminate()
        pool.join()
        raise SystemExit(130) from None
    pool.close()
    pool.join()

    rows = aggregate(done)
    print(f"\nCosts: medians over the seeds docking at least {100 * RELIABLE:.0f} % of the time.")
    column = "budget" if args.budgets else "fuel w"
    extra = "   final fuel weight per seed" if args.budgets else ""
    print(f"{'gamma':>6} {column:>6}   docked per seed        reliable   delta-v   time{extra}")
    for row in rows:
        rates = " ".join(f"{100 * r:5.1f}" for r in row["success_rates"])
        setting = row["fuel_budget"] if args.budgets else row["fuel_weight"]
        weights = ""
        if args.budgets:
            weights = "   " + " ".join(f"{w:5.1f}" for w in row["final_fuel_weights"])
        print(f"{row['gamma']:>6g} {setting:>6g}   {rates:<22s} "
              f"{row['reliable_seeds']}/{row['seeds']}        "
              f"{row['delta_v_median']:5.2f}   {row['time_median']:5.0f} s{weights}")

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps({"runs": done, "configurations": rows}, indent=2))
    baselines_path = Path(args.baselines)
    baselines = json.loads(baselines_path.read_text()) if baselines_path.exists() else None
    if args.budgets:
        plot_budgets(args.plot, done, rows, baselines, args.episodes)
    else:
        plot(args.plot, done, rows, baselines, args.episodes)
    print(f"\nPlot saved to {args.plot}, numbers to {args.results}")


def plot(path, runs, rows, baselines, episodes) -> None:
    """Two panels, one message each, labelled on the points rather than in a legend.

    Left: what the fuel weight does, one line per discount. Right: where the
    best configuration lands against the LQR front and the two-impulse bound.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from orbital_rendezvous.game_view import AMBER, BLUE, GREEN, GRID, MUTED, PANEL, TEXT

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13.0, 5.4), facecolor=PANEL, gridspec_kw={"width_ratios": [1.0, 1.35]}
    )
    for ax in (left, right):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(alpha=0.25, color=GRID)

    # -- left: fuel against the fuel weight, one line per discount ---------------
    gammas = sorted({r["gamma"] for r in rows})
    colours = {g: c for g, c in zip(gammas, (GREEN, AMBER, BLUE, MUTED), strict=False)}
    for gamma in gammas:
        line = [r for r in rows if r["gamma"] == gamma and r["reliable_seeds"]]
        colour = colours.get(gamma, MUTED)
        left.plot([r["fuel_weight"] for r in line], [r["delta_v_median"] for r in line],
                  color=colour, lw=2.2, zorder=2)
        for r in line:
            solid = r["reliable_seeds"] == r["seeds"]
            left.scatter(r["fuel_weight"], r["delta_v_median"], s=90, zorder=3, lw=2,
                         color=colour if solid else PANEL, edgecolor=colour)
            if not solid:
                left.annotate(f"{r['reliable_seeds']}/{r['seeds']} seeds", 
                              (r["fuel_weight"], r["delta_v_median"]), xytext=(0, 11),
                              textcoords="offset points", ha="center", color=MUTED, fontsize=8)
        end = line[-1]
        left.annotate(f"$\\gamma$ = {gamma:g}", (end["fuel_weight"], end["delta_v_median"]),
                      xytext=(10, 0), textcoords="offset points", va="center",
                      color=colour, fontsize=11, fontweight="bold")
    weights = sorted({r["fuel_weight"] for r in rows})
    left.set_xscale("log")
    left.set_xticks(weights)
    left.set_xticklabels([f"{w:g}" for w in weights])
    left.minorticks_off()
    low, high = left.get_ylim()
    left.set_ylim(low - 0.1 * (high - low), high + 0.15 * (high - low))
    left.set_xlim(min(weights) / 1.3, max(weights) * 2.2)
    left.set_xlabel("fuel weight $w_f$", color=TEXT)
    left.set_ylabel(r"fuel spent, median $\Delta v$ [m/s]", color=TEXT)
    left.set_title("A heavier fuel cost only helps with a long horizon", color=TEXT,
                   fontsize=11, loc="left")
    left.text(0.02, 0.03, "hollow: not every seed learned to dock", transform=left.transAxes,
              color=MUTED, fontsize=8)

    # -- right: the best configuration against the classical references ----------
    right.set_title("Against the classical controllers  (lower left is better)",
                    color=TEXT, fontsize=11, loc="left")
    right.set_xlabel("time to dock, median [s]", color=TEXT)
    right.set_ylabel(r"fuel spent, median $\Delta v$ [m/s]", color=TEXT)
    reliable_runs = [r for r in runs if r["summary"]["success_rate"] >= RELIABLE]
    right.scatter([r["summary"]["time_median"] for r in reliable_runs],
                  [r["summary"]["delta_v_median"] for r in reliable_runs],
                  s=16, color=MUTED, alpha=0.45, zorder=2)
    if baselines:
        front = baselines["lqr_front"]
        right.plot([s["time_median"] for s in front], [s["delta_v_median"] for s in front],
                   color=BLUE, lw=2.2, zorder=3)
        at = front[min(1, len(front) - 1)]
        right.annotate("LQR, best trade-offs", (at["time_median"], at["delta_v_median"]),
                       xytext=(10, 8), textcoords="offset points", color=BLUE, fontsize=10)
        bound = baselines["two_impulse"]["delta_v_median"]
        right.axhline(bound, color=TEXT, ls="--", lw=1, zorder=1)
        right.text(0.98, bound, "  ideal two-impulse transfer (not flyable)", color=TEXT,
                   fontsize=9, ha="right", va="bottom", transform=right.get_yaxis_transform())
    complete = [r for r in rows if r["reliable_seeds"] == r["seeds"]]
    if complete and baselines:
        best = min(complete, key=lambda r: r["delta_v_median"])
        agent = baselines["agent"]
        start = (agent["time_median"], agent["delta_v_median"])
        stop = (best["time_median"], best["delta_v_median"])
        right.scatter(*start, s=280, marker="*", color=TEXT, zorder=5)
        right.annotate("default agent", start, xytext=(12, 4), textcoords="offset points",
                       color=TEXT, fontsize=10)
        right.scatter(*stop, s=150, marker="D", color=AMBER, edgecolor=TEXT, zorder=5)
        saving = 100 * (1 - stop[1] / start[1])
        right.annotate(
            f"$\\gamma$ = {best['gamma']:g}, $w_f$ = {best['fuel_weight']:g}\n"
            f"{saving:.0f} % less fuel, all seeds dock",
            stop, xytext=(-30, -115), textcoords="offset points", color=AMBER, fontsize=10,
            arrowprops={"arrowstyle": "-", "color": AMBER, "lw": 0.8},
        )
        right.annotate("", xy=stop, xytext=start,
                       arrowprops={"arrowstyle": "-|>", "color": AMBER, "lw": 1.6,
                                   "shrinkA": 10, "shrinkB": 8})
    slow = max(reliable_runs, key=lambda r: r["summary"]["time_median"], default=None)
    if slow and slow["summary"]["time_median"] > 1.5 * min(
        r["summary"]["time_median"] for r in reliable_runs
    ):
        point = (slow["summary"]["time_median"], slow["summary"]["delta_v_median"])
        right.scatter(*point, s=60, color=MUTED, edgecolor=TEXT, zorder=5)
        right.annotate(f"one run of {len(runs)} found\na slow, cheap approach", point,
                       xytext=(10, 6), textcoords="offset points", color=MUTED, fontsize=9)
    right.text(0.02, 0.03, "grey dots: every other run that docks reliably",
               transform=right.transAxes, color=MUTED, fontsize=8)
    right.set_ylim(bottom=0)

    fig.suptitle(f"Trading time for fuel, on {episodes} unseen starts", color=TEXT,
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PANEL)
    plt.close(fig)


def plot_budgets(path, runs, rows, baselines, episodes) -> None:
    """Two panels: does the agent keep its budget, and where does that put it.

    Left: the fuel spent against the budget asked for, with the line where the
    two are equal. Right: every budget against the LQR front, labelled with its
    budget, as in the fuel-weight study.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from orbital_rendezvous.game_view import AMBER, BLUE, GRID, MUTED, PANEL, TEXT

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13.0, 5.4), facecolor=PANEL, gridspec_kw={"width_ratios": [1.0, 1.35]}
    )
    for ax in (left, right):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(alpha=0.25, color=GRID)

    budgets = sorted(r["fuel_budget"] for r in rows)
    top = max(budgets + [r["delta_v_median"] for r in rows if r["reliable_seeds"]]) * 1.15
    left.plot([0, top], [0, top], color=MUTED, ls="--", lw=1)
    left.text(0.95 * top, 0.95 * top, "spent = budget ", color=MUTED, fontsize=9,
              ha="right", va="bottom")
    for run in runs:
        s = run["summary"]
        reliable = s["success_rate"] >= RELIABLE
        left.scatter(run["fuel_budget"], s["delta_v_median"] if reliable else 0.02 * top,
                     s=40, color=AMBER if reliable else PANEL, edgecolor=AMBER, lw=1.5,
                     alpha=0.9, zorder=3, marker="o" if reliable else "v")
    left.set_xlim(0, top)
    left.set_ylim(0, top)
    left.set_xlabel("fuel budget asked for [m/s]", color=TEXT)
    left.set_ylabel(r"fuel spent, median $\Delta v$ [m/s]", color=TEXT)
    left.set_title("Does the agent keep its budget?", color=TEXT, fontsize=11, loc="left")
    left.text(0.98, 0.03, "one dot per seed; triangles on the axis: seeds that did not dock",
              transform=left.transAxes, color=MUTED, fontsize=8, ha="right")
    weights = [w for r in rows for w in r["final_fuel_weights"] if w is not None]
    missed = all(r["delta_v_median"] > r["fuel_budget"] for r in rows if r["reliable_seeds"])
    if weights and missed:
        left.text(0.04, 0.55, f"above the line: budget missed\nwhile the price of fuel rose\n"
                  f"to {min(weights):.0f}–{max(weights):.0f}", transform=left.transAxes,
                  color=AMBER, fontsize=10, va="center")

    right.set_title("Against the classical controllers  (lower left is better)",
                    color=TEXT, fontsize=11, loc="left")
    right.set_xlabel("time to dock, median [s]", color=TEXT)
    right.set_ylabel(r"fuel spent, median $\Delta v$ [m/s]", color=TEXT)
    if baselines:
        front = baselines["lqr_front"]
        right.plot([s["time_median"] for s in front], [s["delta_v_median"] for s in front],
                   color=BLUE, lw=2.2, zorder=3)
        at = front[min(1, len(front) - 1)]
        right.annotate("LQR, best trade-offs", (at["time_median"], at["delta_v_median"]),
                       xytext=(10, 8), textcoords="offset points", color=BLUE, fontsize=10)
        bound = baselines["two_impulse"]["delta_v_median"]
        right.axhline(bound, color=TEXT, ls="--", lw=1, zorder=1)
        right.text(0.98, bound, "  ideal two-impulse transfer (not flyable)", color=TEXT,
                   fontsize=9, ha="right", va="bottom", transform=right.get_yaxis_transform())
        agent = baselines["agent"]
        right.scatter(agent["time_median"], agent["delta_v_median"], s=280, marker="*",
                      color=TEXT, zorder=5)
        right.annotate("default agent", (agent["time_median"], agent["delta_v_median"]),
                       xytext=(12, 4), textcoords="offset points", color=TEXT, fontsize=10)
    shown = [r for r in sorted(rows, key=lambda r: -r["fuel_budget"]) if r["reliable_seeds"]]
    for i, row in enumerate(shown):
        point = (row["time_median"], row["delta_v_median"])
        right.scatter(*point, s=110, marker="D", color=AMBER, edgecolor=TEXT, zorder=5)
        # Alternate the labels below and to the right, so that neighbours never touch.
        right.annotate(f"budget {row['fuel_budget']:g} m/s  ({row['reliable_seeds']}/"
                       f"{row['seeds']} seeds)", point, xytext=(6, -120 - 18 * i),
                       textcoords="offset points", color=AMBER, fontsize=9,
                       arrowprops={"arrowstyle": "-", "color": AMBER, "lw": 0.6})
    right.set_ylim(bottom=0)

    fig.suptitle(f"A Lagrange multiplier on fuel, on {episodes} unseen starts", color=TEXT,
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PANEL)
    plt.close(fig)


if __name__ == "__main__":
    main()
