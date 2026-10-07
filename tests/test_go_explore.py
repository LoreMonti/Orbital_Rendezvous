"""Tests for Go-Explore's phase 1.

A docking found by returning to saved states is worth something only if the
thrusts it keeps fly the same way again from the start; so a found docking is
replayed from a fresh reset. The weights decide where the search goes next,
and a sign error there would send it away from the port and still run, so
they are pinned too.
"""

from dataclasses import replace

import numpy as np
import pytest

from orbital_rendezvous import EnvConfig, Outcome, RendezvousEnv
from orbital_rendezvous.go_explore import (
    GoExploreConfig,
    cell,
    cell_size,
    explore,
    replay,
    weights,
)

STRICT = EnvConfig(keep_out_radius=20.0, approach_cone_deg=15.0)
SETTINGS = GoExploreConfig()


def near_the_port() -> RendezvousEnv:
    env = RendezvousEnv(STRICT)
    env.set_start_region(4.0, 8.0, 10.0)
    return env


def test_cells_group_nearby_states_and_split_speeds():
    # Out here cells are 4 m: 0.5 m doubled up to the largest size within 5 m.
    a = cell(np.array([41.0, 42.0, 0.0, 0.0]), SETTINGS)
    assert a == cell(np.array([43.9, 43.9, 0.01, 0.0]), SETTINGS)
    assert a != cell(np.array([44.1, 42.0, 0.0, 0.0]), SETTINGS)        # the next square
    assert a != cell(np.array([41.0, 42.0, 0.3, 0.0]), SETTINGS)        # faster


def test_cells_shrink_next_to_the_port():
    assert cell_size(100.0, SETTINGS) == 5.0
    assert cell_size(4.0, SETTINGS) == 1.0
    assert cell_size(0.5, SETTINGS) == 0.5
    # 1 m and 3 m from the port are different cells; on a 5 m grid they were one.
    assert cell(np.array([0.0, 1.0, 0.0, 0.0]), SETTINGS) != cell(np.array([0.0, 3.0, 0.0, 0.0]),
                                                                  SETTINGS)


def test_the_search_returns_to_cells_near_the_port_and_seldom_chosen():
    p = weights(np.array([10.0, 100.0, 10.0]), np.array([0.0, 0.0, 8.0]), SETTINGS)
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[2] > 0.0           # chosen more often, chosen less
    assert p[0] > p[1] > 0.0           # further from the port, chosen less


def test_a_docking_found_flies_again_from_the_start():
    env = near_the_port()
    found = explore(env, 3, replace(SETTINGS, budget=20_000, after_docking=500))
    assert found.docked
    assert np.all(np.linalg.norm(found.actions, axis=1) <= 1.0 + 1e-12)
    check = replay(near_the_port(), 3, found.actions)
    assert check["outcome"] is Outcome.DOCKED and not check["violated"]
    assert found.rounds <= found.first_docking + 500            # stops after the margin


def test_without_a_docking_the_archive_still_reports_how_close_it_came():
    env = RendezvousEnv(STRICT)
    found = explore(env, 1, replace(SETTINGS, budget=200))
    assert not found.docked and len(found.actions) == 0
    assert found.cells > 1 and found.closest <= np.hypot(*found.start[:2])


def test_mirrored_environments_are_refused():
    with pytest.raises(ValueError):
        explore(RendezvousEnv(replace(STRICT, mirror_symmetry=True)), 0)
