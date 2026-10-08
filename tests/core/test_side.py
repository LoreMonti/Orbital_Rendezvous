"""Tests for the side chosen by the agent, through the mirror.

The choice must be read once and only once: if a_s kept acting after the
first step, the agent could switch sides mid-manoeuvre, which is the very
average this option exists to remove; and if the first step were flown in the
chosen frame, the agent would steer by a state it never saw.
"""

from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from orbital_rendezvous import EnvConfig, RendezvousEnv

MIRROR = EnvConfig(keep_out_radius=20.0, approach_cone_deg=15.0, mirror_symmetry=True,
                   initial_radius_range=(28.0, 40.0), start_angle_range_deg=(35.0, 45.0))
SIDE = replace(MIRROR, side_choice=True)


def test_side_choice_needs_the_mirror():
    with pytest.raises(ValueError):
        RendezvousEnv(EnvConfig(side_choice=True))


def test_the_action_gains_one_component():
    assert RendezvousEnv(MIRROR).action_space.shape == (2,)
    assert RendezvousEnv(SIDE).action_space.shape == (3,)
    assert RendezvousEnv(replace(SIDE, engine_switch=True)).action_space.shape == (4,)


def test_choosing_by_the_sign_of_x_reproduces_the_mirror_of_step_23():
    # The rule of Step 23, taken as the agent's choice, with no thrust on the
    # first step (the only step flown differently): the same flight after it.
    fixed, chosen = RendezvousEnv(MIRROR), RendezvousEnv(SIDE)
    rng = np.random.default_rng(0)
    for seed in range(10):
        fixed.reset(seed=seed)
        chosen.reset(seed=seed)
        a_s = -1.0 if chosen.state[0] > 0.0 else 1.0
        of, *_ = fixed.step(np.zeros(2))
        oc, *_ = chosen.step(np.array([0.0, 0.0, a_s]))
        np.testing.assert_array_equal(of, oc)
        for _ in range(20):
            thrust = rng.uniform(-1.0, 1.0, 2)
            of, rf, df, *_ = fixed.step(thrust)
            oc, rc, dc, *_ = chosen.step(np.append(thrust, rng.uniform(-1.0, 1.0)))
            np.testing.assert_array_equal(of, oc)
            assert rf == rc and df == dc
            if df:
                break


@pytest.mark.parametrize("a_s, mirror", [(0.3, 1.0), (0.0, 1.0), (-0.2, -1.0)])
def test_the_first_step_sets_the_mirror_and_later_steps_cannot(a_s, mirror):
    env = RendezvousEnv(SIDE)
    obs, _ = env.reset(seed=0)
    assert env.mirror == 1.0                         # the start is seen as it is
    np.testing.assert_allclose(obs[:4], env.state / env._obs_scale, rtol=1e-6)
    obs, *_ , info = env.step(np.array([0.0, 0.0, a_s]))
    assert env.mirror == mirror and info["mirror"] == mirror
    np.testing.assert_allclose(obs[[0, 2]] * mirror, (env.state / env._obs_scale)[[0, 2]],
                               rtol=1e-6)
    for later in (1.0, -1.0, 0.5):
        env.step(np.array([0.0, 0.0, later]))
        assert env.mirror == mirror                  # kept for the whole episode


def test_the_first_step_is_flown_as_it_is_the_next_ones_in_the_chosen_frame():
    env = RendezvousEnv(SIDE)
    env.reset(seed=0)
    *_, info = env.step(np.array([0.5, 0.0, -1.0]))      # chooses the mirror
    assert info["thrust"][0] == pytest.approx(0.5 * SIDE.max_thrust)
    *_, info = env.step(np.array([0.5, 0.0, 1.0]))
    assert info["thrust"][0] == pytest.approx(-0.5 * SIDE.max_thrust)


def test_inside_the_sphere_there_is_no_choice():
    near = replace(SIDE, initial_radius_range=(2.0, 19.0), start_angle_range_deg=(0.0, 10.0))
    env = RendezvousEnv(near)
    for seed in range(10):
        env.reset(seed=seed)
        env.step(np.array([0.0, 0.0, -1.0]))
        assert env.mirror == 1.0


def test_the_choice_is_made_again_in_every_episode():
    env = RendezvousEnv(SIDE)
    env.reset(seed=0)
    env.step(np.array([0.0, 0.0, -1.0]))
    assert env.mirror == -1.0
    env.reset(seed=1)
    assert env.mirror == 1.0 and env.side_open
    env.step(np.array([0.0, 0.0, 1.0]))
    assert env.mirror == 1.0


def test_check_env_passes_with_the_side_choice():
    check_env(RendezvousEnv(SIDE), skip_render_check=True)
