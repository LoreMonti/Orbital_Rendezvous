"""Tests for the learned value of the planner.

The returns are the labels the network learns from, so a wrong discount or an
off-by-one would teach a wrong value and still train smoothly; they are
checked by hand. The safeguards of `LearnedValue`, the zone where the network
is off and the penalty-only correction, are what kept the planner from
hovering at the port and from flying off to invented values, so they are
pinned too.
"""

from dataclasses import replace

import numpy as np
import pytest

from orbital_rendezvous import EnvConfig, RendezvousEnv
from orbital_rendezvous.planning.mpc import PlannerConfig, SamplingPlanner
from orbital_rendezvous.planning.value import (
    LearnedValue,
    ValueNet,
    discounted_returns,
    observation,
    prior,
)

STRICT = EnvConfig(keep_out_radius=20.0, approach_cone_deg=15.0)


def test_returns_by_hand():
    rewards = np.array([-1.0, -1.0, 100.0])
    np.testing.assert_allclose(discounted_returns(rewards, 0.5), [-1.0 - 0.5 + 25.0, -1.0 + 50.0,
                                                                   100.0])
    assert discounted_returns(np.array([7.0]), 0.99)[0] == 7.0


def test_before_any_data_the_value_is_the_prior():
    env = RendezvousEnv(STRICT)
    states = np.array([[60.0, -80.0, 0.0, 0.0], [0.0, 30.0, 0.0, -0.1]])
    value = LearnedValue(env, ValueNet(), cone_zone=False, penalty_only=False)
    np.testing.assert_allclose(value(states, 10), prior(env, states))
    np.testing.assert_allclose(prior(env, states), -20.0 * np.hypot(*states[:, :2].T) / 500.0)


def test_the_network_fits_a_known_function():
    rng = np.random.default_rng(0)
    obs = rng.uniform(-1.0, 1.0, size=(2000, 5))
    target = 30.0 * obs[:, 0] - 20.0 * obs[:, 1] ** 2
    net = ValueNet(seed=0)
    error = net.fit(obs, target, steps=3000, lr=3e-3)
    assert error < 4.0                                      # an RMS error of 2 on a range of 80
    np.testing.assert_allclose(net.correction(obs[:5]), target[:5], atol=6.0)


def test_the_network_is_off_down_the_cone_near_the_port_and_on_beside_the_sphere():
    env = RendezvousEnv(STRICT)
    value = LearnedValue(env, ValueNet())
    down_the_cone = np.array([[0.0, 15.0, 0.0, 0.0]])
    beside_the_sphere = np.array([[21.0, 7.0, 0.0, 0.0]])       # 22 m out, 72 degrees off axis
    far = np.array([[0.0, 60.0, 0.0, 0.0]])
    assert value.weight(down_the_cone)[0] == 0.0
    assert value.weight(beside_the_sphere)[0] == 1.0
    assert value.weight(far)[0] == 1.0
    radial = LearnedValue(env, ValueNet(), cone_zone=False)
    assert radial.weight(beside_the_sphere)[0] == 0.0           # the zone that let it stall


def test_a_penalty_only_correction_never_raises_the_prior():
    env = RendezvousEnv(STRICT)
    net = ValueNet(seed=1)
    rng = np.random.default_rng(1)
    states = np.column_stack([rng.uniform(-200, 200, (500, 2)), rng.uniform(-0.2, 0.2, (500, 2))])
    obs = observation(env, states, 50)
    net.fit(obs, rng.normal(0.0, 50.0, 500), steps=200)        # an arbitrary, noisy network
    value = LearnedValue(env, net)
    assert np.all(value(states, 50) <= prior(env, states) + 1e-9)


def test_an_ensemble_lowers_the_value_where_its_networks_disagree():
    env = RendezvousEnv(STRICT)
    a, b = ValueNet(seed=0), ValueNet(seed=1)
    obs = np.random.default_rng(2).uniform(-1, 1, (300, 5))
    a.fit(obs, np.full(300, 40.0), steps=300)
    b.fit(obs, np.full(300, -40.0), steps=300)
    states = np.array([[0.0, 100.0, 0.0, 0.0]])
    mean_only = LearnedValue(env, [a, b], beta=0.0, penalty_only=False)
    cautious = LearnedValue(env, [a, b], beta=1.0, penalty_only=False)
    assert cautious(states, 0)[0] < mean_only(states, 0)[0] - 20.0


def test_the_planner_uses_the_learned_value_and_needs_it():
    env = RendezvousEnv(STRICT)
    with pytest.raises(ValueError):
        SamplingPlanner(env, PlannerConfig(value="learned"))
    calls = []

    def value_fn(states, steps):
        calls.append(steps)
        return np.zeros(len(states))

    planner = SamplingPlanner(env, PlannerConfig(value="learned", horizon=5, block=1,
                                                 samples=8, elites=2, iterations=1),
                              value_fn=value_fn)
    planner.score(np.array([0.0, 100.0, 0.0, 0.0]), np.zeros((8, 5, 2)), steps_done=7)
    assert calls == [12]                                        # valued after 7 + 5 steps


def test_mirrored_environments_are_still_refused():
    with pytest.raises(ValueError):
        SamplingPlanner(RendezvousEnv(replace(STRICT, mirror_symmetry=True)))
