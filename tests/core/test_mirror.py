"""Tests for the mirror symmetry of the oriented target.

The mirror must be invisible to the physics and exact for the agent: the
chaser flies the true Clohessy-Wiltshire dynamics, while the agent sees and
commands a state on the side x <= 0 when it starts outside the keep-out
sphere. A wrong sign would make the agent push the chaser away from the cone
on half of the starts, and train anyway.
"""

from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from orbital_rendezvous import EnvConfig, RendezvousEnv

MIRROR = EnvConfig(keep_out_radius=20.0, approach_cone_deg=15.0, mirror_symmetry=True,
                   initial_radius_range=(28.0, 40.0), start_angle_range_deg=(35.0, 45.0))


def starts(config, n=40):
    env = RendezvousEnv(config)
    for seed in range(n):
        obs, _ = env.reset(seed=seed)
        yield env, obs


def test_mirror_is_off_by_default_and_changes_nothing():
    plain = RendezvousEnv(EnvConfig())
    obs, _ = plain.reset(seed=3)
    assert obs.shape == (5,) and plain.mirror == 1.0
    np.testing.assert_allclose(obs[:4], plain.state / plain._obs_scale, rtol=1e-6)


def test_every_start_outside_the_sphere_is_shown_on_the_side_x_le_0():
    sides = set()
    for env, obs in starts(MIRROR):
        assert obs.shape == (5,) and env.observation_space.contains(obs)
        assert obs[0] <= 0.0
        true_side = np.sign(env.state[0])
        assert env.mirror == (-1.0 if true_side > 0 else 1.0)
        # Shown x and vx are the true ones, flipped on the mirrored side only.
        np.testing.assert_allclose(obs[[0, 2]] * env.mirror,
                                   (env.state / env._obs_scale)[[0, 2]], rtol=1e-6)
        np.testing.assert_allclose(obs[[1, 3]], (env.state / env._obs_scale)[[1, 3]], rtol=1e-6)
        sides.add(true_side)
    assert sides == {-1.0, 1.0}   # both sides were drawn


def test_starts_inside_the_keep_out_sphere_are_never_mirrored():
    near = EnvConfig(keep_out_radius=20.0, approach_cone_deg=15.0, mirror_symmetry=True,
                     initial_radius_range=(2.0, 19.0), start_angle_range_deg=(0.0, 10.0))
    sides = set()
    for env, obs in starts(near):
        assert env.mirror == 1.0
        np.testing.assert_allclose(obs[:4], env.state / env._obs_scale, rtol=1e-6)
        sides.add(np.sign(env.state[0]))
    assert sides == {-1.0, 1.0}   # starts on both sides, none of them mirrored


@pytest.mark.parametrize("start_x", [-30.0, 30.0])
def test_the_radial_command_is_flipped_back_on_the_mirrored_side(start_x):
    env = RendezvousEnv(MIRROR)
    env.reset(seed=0)
    env.state = np.array([start_x, 30.0, 0.0, 0.0])
    env.mirror = -1.0 if start_x > 0 else 1.0
    _, _, _, _, info = env.step(np.array([0.5, -0.25]))
    # The agent asks to move towards the axis (+x in its frame, as on the
    # side x <= 0); the true thrust must point towards the axis too.
    assert info["thrust"][0] * np.sign(start_x) < 0.0
    assert info["thrust"][0] == pytest.approx(-0.5 * np.sign(start_x) * MIRROR.max_thrust)
    assert info["thrust"][1] == pytest.approx(-0.25 * MIRROR.max_thrust)


def test_two_mirrored_starts_see_the_same_state():
    a, b = RendezvousEnv(MIRROR), RendezvousEnv(MIRROR)
    a.reset(seed=0)
    b.reset(seed=0)
    a.state, a.mirror = np.array([-25.0, 25.0, 0.01, -0.02]), 1.0
    b.state, b.mirror = np.array([25.0, 25.0, -0.01, -0.02]), -1.0
    oa, ob = a._observation(), b._observation()
    np.testing.assert_allclose(oa, ob)


def test_the_side_is_kept_when_the_chaser_crosses_the_axis():
    env = RendezvousEnv(MIRROR)
    env.reset(seed=0)
    env.state, env.mirror = np.array([2.0, 30.0, -1.0, 0.0]), -1.0   # crossing to x < 0
    obs, *_ = env.step(np.zeros(2))
    assert env.state[0] < 0.0 and env.mirror == -1.0
    assert obs[0] > 0.0                        # now shown on the far side, not jumped back


def test_inside_the_sphere_the_mirror_changes_nothing():
    # Same seeds, same actions: observations and rewards bit for bit, so the
    # first stages of a curriculum train exactly as without the mirror.
    plain = EnvConfig(keep_out_radius=20.0, approach_cone_deg=15.0,
                      initial_radius_range=(2.0, 19.0), start_angle_range_deg=(0.0, 10.0))
    a, b = RendezvousEnv(plain), RendezvousEnv(replace(plain, mirror_symmetry=True))
    rng = np.random.default_rng(0)
    for seed in range(10):
        np.testing.assert_array_equal(a.reset(seed=seed)[0], b.reset(seed=seed)[0])
        for _ in range(50):
            action = rng.uniform(-1.0, 1.0, 2)
            oa, ra, da, *_ = a.step(action)
            ob, rb, db, *_ = b.step(action)
            np.testing.assert_array_equal(oa, ob)
            assert ra == rb and da == db
            if da:
                break


def test_check_env_passes_with_the_mirror():
    check_env(RendezvousEnv(MIRROR), skip_render_check=True)
