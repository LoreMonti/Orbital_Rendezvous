"""Tests for the graph of exact manoeuvres (Step 28).

The graph is only as good as its edges: a transfer that missed its node, or an
arc through the keep-out sphere counted as legal, would give values to ways in
that do not exist, and the planner would follow them. So an edge is flown with
the exact propagation, the rule is checked on a manoeuvre known to break it,
and the value is checked where its shape is known: zero distance, nothing to
pay; further, less; behind the station, round one side.
"""

from dataclasses import replace

import numpy as np
import pytest

from orbital_rendezvous import EnvConfig, RendezvousEnv
from orbital_rendezvous.core.dynamics import state_transition
from orbital_rendezvous.planning.graph import (
    CWGraph,
    GraphConfig,
    GraphValue,
    impulse_limit,
    legal,
    transfer,
)

STRICT = EnvConfig(keep_out_radius=20.0, approach_cone_deg=15.0)


@pytest.fixture(scope="module")
def graph():
    return CWGraph(RendezvousEnv(STRICT))


def test_a_transfer_lands_on_its_node():
    env = RendezvousEnv(STRICT)
    p0, v0, p1 = np.array([[60.0, -40.0]]), np.array([[0.01, 0.02]]), np.array([[30.0, 25.0]])
    for t in (100.0, 400.0):
        depart, dv1, arrive = transfer(env.n, t, p0, v0, p1)
        end = state_transition(env.n, t) @ np.append(p0[0], depart[0])
        np.testing.assert_allclose(end[:2], p1[0], atol=1e-9)
        np.testing.assert_allclose(end[2:], arrive[0], atol=1e-12)
        np.testing.assert_allclose(dv1[0], depart[0] - v0[0])


def test_an_arc_through_the_sphere_outside_the_cone_is_illegal():
    env = RendezvousEnv(STRICT)
    # From one side of the station to the other, straight through it.
    p0, p1 = np.array([[-40.0, 0.0]]), np.array([[40.0, 0.0]])
    depart, _, _ = transfer(env.n, 400.0, p0, np.zeros((1, 2)), p1)
    assert not legal(env.n, 400.0, p0, depart, 30, 20.0, 15.0, 1.0)[0]
    # Down the cone to the port, briefly: legal. (Over 400 s the Coriolis term
    # bows the same arc out of the cone, which is the corridor's difficulty.)
    p0, p1 = np.array([[0.0, 15.0]]), np.zeros((1, 2))
    depart, _, _ = transfer(env.n, 100.0, p0, np.zeros((1, 2)), p1)
    assert legal(env.n, 100.0, p0, depart, 30, 20.0, 15.0, 1.0)[0]


def test_an_impulse_must_fit_in_a_share_of_the_flight():
    env = RendezvousEnv(STRICT)
    config = GraphConfig()
    assert impulse_limit(config, env, 50.0) == pytest.approx(0.25 * 0.002 * 50.0)
    assert impulse_limit(config, env, 10_000.0) == config.dv_max


def test_the_port_is_worth_the_bonus_and_every_node_less(graph):
    np.testing.assert_array_equal(graph.points[0], [0.0, 0.0])
    assert graph.value[0] == pytest.approx(100.0)
    assert np.all(np.isfinite(graph.dv_to_go))                  # every node has a way in
    assert np.all(graph.value[1:] < 100.0)
    # Along one direction, further out is worth less: all the way down the axis,
    # and beyond 60 m in every direction. Beside the sphere it need not be: a
    # node 25 m out at 90 degrees must back away to reach the mouth of the cone,
    # and is worth less than one 60 m out, which reaches it in one manoeuvre.
    def along(direction, radii):
        u = np.array([np.sin(np.radians(direction)), np.cos(np.radians(direction))])
        return graph.value[[int(np.argmin(np.linalg.norm(graph.points - r * u, axis=1)))
                            for r in radii]]
    assert np.all(np.diff(along(0.0, (25.0, 60.0, 110.0, 200.0))) < 0.0)
    for direction in (90.0, 180.0):
        assert np.all(np.diff(along(direction, (60.0, 110.0, 200.0))) < 0.0)
    assert along(90.0, (25.0,))[0] < along(90.0, (60.0,))[0]


def test_from_behind_the_station_the_cheapest_way_goes_round_one_side(graph):
    k = int(np.argmin(np.linalg.norm(graph.points - np.array([0.0, -110.0]), axis=1)))
    way = graph.points[graph.path(k)]
    assert graph.path(k)[-1] == 0
    sides = np.sign(way[1:-1, 0])
    assert np.all(sides[sides != 0] == sides[sides != 0][0])     # one side, kept
    # Every node on the way is outside the sphere, or inside it within the cone.
    r = np.hypot(*way[1:-1].T)
    in_cone = way[1:-1, 1] >= r * np.cos(np.radians(15.0))
    assert np.all((r >= 20.0) | in_cone)


def test_the_value_grows_down_the_cone_towards_the_port(graph):
    value = GraphValue(graph)
    on_axis = np.array([[0.0, y, 0.0, 0.0] for y in (30.0, 20.0, 10.0, 5.0, 2.0)])
    v = value(on_axis, 50)
    assert np.all(np.diff(v) > 0.0)                            # no plateau on the way in


def test_a_state_with_no_legal_way_gets_the_fallback(graph):
    value = GraphValue(graph)
    lost = np.array([[0.0, 1000.0, 0.0, 0.0]])                  # far beyond every node's reach
    assert value(lost, 0)[0] == -100.0


def test_the_pilot_docks_from_behind_the_station_round_one_side(graph):
    from orbital_rendezvous.core.evaluation import rollout
    from orbital_rendezvous.core.rewards import Outcome
    from orbital_rendezvous.planning.graph import GraphPilot

    env = RendezvousEnv(replace(STRICT, start_angle_range_deg=(170.0, 180.0)))
    pilot = GraphPilot(env, graph, seed=0)
    with np.errstate(all="ignore"):
        run = rollout(env, pilot, 3)
    assert run.outcome is Outcome.DOCKED and not run.violated
    sides = np.sign(run.positions[np.hypot(*run.positions.T) > 25.0][:, 0])
    assert abs(sides.mean()) > 0.8                              # round one side, not both


def test_the_pilot_plans_with_a_margin_on_the_docking_speed(graph):
    from orbital_rendezvous.planning.graph import GraphPilot

    env = RendezvousEnv(STRICT)
    pilot = GraphPilot(env, graph, docking_margin=0.8)
    assert pilot.planner.env.config.docking_speed == pytest.approx(0.8 * STRICT.docking_speed)
    assert env.config.docking_speed == STRICT.docking_speed      # the rule itself unchanged


def test_the_descent_starts_only_inside_the_cone(graph):
    from orbital_rendezvous.planning.graph import GraphPilot

    pilot = GraphPilot(RendezvousEnv(STRICT), graph)
    assert pilot._inside_cone(np.array([0.0, 10.0, 0.0, 0.0]))
    assert not pilot._inside_cone(np.array([0.0, 25.0, 0.0, 0.0]))   # mouth, outside the sphere
    assert not pilot._inside_cone(np.array([8.0, 8.0, 0.0, 0.0]))    # inside, off the cone
