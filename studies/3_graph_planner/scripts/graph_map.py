"""Draw the graph of exact manoeuvres and its cheapest way in from two starts (Step 28).

The nodes are drawn grey, and the way in from each start in blue, node by node,
along the real Clohessy-Wiltshire arcs of the transfers, not straight lines: a
straight line between two nodes may cross the keep-out sphere where the arc
does not.

Usage:
    python studies/3_graph_planner/scripts/graph_map.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from orbital_rendezvous import RendezvousEnv
from orbital_rendezvous.core.utils import build_configs, load_config
from orbital_rendezvous.planning.graph import CWGraph, blocks, impulse_limit, legal, transfer

STARTS = ((110.0, 170.0, "Start behind the station (110 m, 170°)"),
          (150.0, 120.0, "Start to the side (150 m, 120°)"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="studies/2_oriented_port/configs/ppo_corridor.yaml")
    parser.add_argument("--plot", default="studies/3_graph_planner/assets/graph_map.png")
    return parser.parse_args()


def arc(graph: CWGraph, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """The points of the cheapest legal transfer from node ``a`` to node ``b``, and its time."""
    env, cfg, ec = graph.env, graph.config, graph.env.config
    best = None
    for t in cfg.times:
        depart, dv1, arrive = transfer(env.n, t, a[None], np.zeros((1, 2)), b[None])
        d1, d2 = np.linalg.norm(dv1), np.linalg.norm(arrive)
        limit = impulse_limit(cfg, env, t)
        if d1 <= limit and d2 <= limit and legal(env.n, t, a[None], depart, cfg.samples,
                                                 ec.keep_out_radius, ec.approach_cone_deg,
                                                 ec.docking_radius)[0]:
            cost = d1 + d2 + cfg.time_weight * t
            if best is None or cost < best[0]:
                best = (cost, t, depart[0])
    _, t, depart = best
    points = []
    for s in np.linspace(0.0, t, 60):
        rr, rv, _, _ = blocks(env.n, s)
        with np.errstate(all="ignore"):
            points.append(a @ rr.T + depart @ rv.T)
    return np.array(points), t


def main() -> None:
    args = parse_args()
    env_config, reward_config = build_configs(load_config(args.config))
    graph = CWGraph(RendezvousEnv(env_config, reward_config))
    points = graph.points
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
    cone = np.radians(env_config.approach_cone_deg)
    for ax, (r0, a0, title) in zip(axes, STARTS, strict=True):
        ax.add_patch(plt.Circle((0, 0), env_config.keep_out_radius, color="0.85", zorder=0))
        ax.fill([0, -95 * np.sin(cone), 95 * np.sin(cone)], [0, 95 * np.cos(cone),
                95 * np.cos(cone)], color="#dff0d8", zorder=0)
        ax.scatter(points[1:, 0], points[1:, 1], s=7, color="0.65", zorder=1)
        start = r0 * np.array([np.sin(np.radians(a0)), np.cos(np.radians(a0))])
        way = graph.path(int(np.argmin(np.linalg.norm(points - start, axis=1))))
        for q0, q1 in zip(way[:-1], way[1:], strict=True):
            line, t = arc(graph, points[q0], points[q1])
            ax.plot(line[:, 0], line[:, 1], "-", color="tab:blue", lw=2.2, zorder=2)
            middle = line[len(line) // 2]
            if np.hypot(*middle) > 35.0:
                ax.annotate(f"{t:.0f} s", middle, xytext=(6, -4), textcoords="offset points",
                            fontsize=8, color="tab:blue")
        for n, q in enumerate(way):
            point = points[q]
            ax.plot(*point, "o", color="tab:red" if q == 0 else "tab:blue",
                    ms=11 if q == 0 else 8, zorder=3)
            if q:
                ax.annotate(str(n + 1), point, xytext=(-14 if point[0] < 0 else 6, 5),
                            textcoords="offset points", fontsize=11, weight="bold",
                            color="tab:blue")
        ax.annotate("port", (0, 0), xytext=(10, -14), textcoords="offset points", fontsize=10,
                    weight="bold", color="tab:red")
        ax.text(0, -28, "keep-out sphere", ha="center", fontsize=8, color="0.35")
        ax.text(0, 80, "cone", ha="center", fontsize=9, color="#3c763d")
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.set_xlim(-130, 130)
        ax.set_ylim(-130, 95)
        ax.set_xlabel("x, radial (m)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("y, along track (m)")
    fig.suptitle("The graph as a map: fixed nodes (grey) and the cheapest way in (blue, real "
                 "CW arcs, numbered in order)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=110)
    print(f"Saved {args.plot}")


if __name__ == "__main__":
    main()
