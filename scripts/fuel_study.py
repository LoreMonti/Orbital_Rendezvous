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
    parser.add_argument("--results", default="assets/fuel_study.json")
    parser.add_argument("--plot", default="assets/fuel_study.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    jobs = make_jobs(args.gammas, args.fuel_weights, args.seeds, args.timesteps,
                     args.episodes, args.directory, args.curriculum_start, args.curriculum_ramp,
                     args.deadzone)
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
    print(f"{'gamma':>6} {'fuel w':>6}   docked per seed        reliable   delta-v   time")
    for row in rows:
        rates = " ".join(f"{100 * r:5.1f}" for r in row["success_rates"])
        print(f"{row['gamma']:>6g} {row['fuel_weight']:>6g}   {rates:<22s} "
              f"{row['reliable_seeds']}/{row['seeds']}        "
              f"{row['delta_v_median']:5.2f}   {row['time_median']:5.0f} s")

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps({"runs": done, "configurations": rows}, indent=2))
    baselines_path = Path(args.baselines)
    baselines = json.loads(baselines_path.read_text()) if baselines_path.exists() else None
    plot(args.plot, done, rows, baselines, args.episodes)
    print(f"\nPlot saved to {args.plot}, numbers to {args.results}")


def plot(path, runs, rows, baselines, episodes) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from orbital_rendezvous.game_view import AMBER, BLUE, GREEN, GRID, MUTED, ORANGE, PANEL, TEXT

    # Palettes repeat rather than silently drop values beyond the third.
    palette, shapes = (GREEN, AMBER, ORANGE), ("o", "s", "D", "^")
    gammas = sorted({r["gamma"] for r in rows})
    weights = sorted({r["fuel_weight"] for r in rows})
    colours = {g: palette[i % len(palette)] for i, g in enumerate(gammas)}
    markers = {w: shapes[i % len(shapes)] for i, w in enumerate(weights)}

    fig, ax = plt.subplots(figsize=(8.6, 5.6), facecolor=PANEL)
    ax.set_facecolor(PANEL)
    if baselines:
        front = baselines["lqr_front"]
        ax.plot([s["time_median"] for s in front], [s["delta_v_median"] for s in front],
                color=BLUE, lw=2, marker="o", ms=3, label="LQR, best trade-offs")
        ax.axhline(baselines["two_impulse"]["delta_v_median"], color=TEXT, ls="--", lw=1,
                   label="ideal two-impulse transfer (not flyable)")
        agent = baselines["agent"]
        ax.scatter([agent["time_median"]], [agent["delta_v_median"]], s=260, marker="*",
                   color=TEXT, zorder=6, label="default agent")
    # Every seed faintly, and the median of each configuration boldly.
    for run in runs:
        s = run["summary"]
        if s["success_rate"] >= RELIABLE:
            ax.scatter(s["time_median"], s["delta_v_median"], s=18, alpha=0.35,
                       color=colours[run["gamma"]], marker=markers[run["fuel_weight"]])
    for row in rows:
        if not row["reliable_seeds"]:
            continue
        ax.scatter(row["time_median"], row["delta_v_median"], s=110, edgecolor=TEXT, lw=0.8,
                   color=colours[row["gamma"]], marker=markers[row["fuel_weight"]], zorder=5,
                   label=f"PPO, $\\gamma$ = {row['gamma']:g}, $w_f$ = {row['fuel_weight']:g}"
                         f"  ({row['reliable_seeds']}/{row['seeds']} seeds dock)")
    ax.set_xlabel("time to dock, median [s]", color=TEXT)
    ax.set_ylabel(r"fuel spent, median $\Delta v$ [m/s]", color=TEXT)
    ax.set_title(f"Trading time for fuel, on {episodes} unseen starts  (lower left is better)",
                 color=TEXT, fontsize=11)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(alpha=0.25, color=GRID)
    ax.set_ylim(bottom=0)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PANEL)
    plt.close(fig)


if __name__ == "__main__":
    main()
