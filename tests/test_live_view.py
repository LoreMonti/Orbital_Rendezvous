"""Tests for the training window and the callback that feeds it.

The callback is driven by hand with episodes whose outcome, return and delta-v
are known in advance, so its bookkeeping is checked exactly. A short real PPO
run then checks that the pieces fit together inside Stable-Baselines3.
"""

import numpy as np
import pytest

from orbital_rendezvous import EnvConfig, Outcome, RendezvousEnv
from orbital_rendezvous.callbacks import LiveViewCallback
from orbital_rendezvous.live_view import LiveView, TrainingCurves, rolling_mean


class RecordingView:
    """Stands in for LiveView and remembers what it was asked to draw."""

    def __init__(self):
        self.episodes = []
        self.curve_updates = 0

    def show_episode(self, positions, velocities, thrusts, outcome, episode):
        assert len(positions) == len(velocities) == len(thrusts)
        self.episodes.append((len(positions), outcome, episode))

    def update_curves(self, curves):
        self.curve_updates += 1


def step_info(outcome=None, shaping=5.0, fuel=-0.02, terminal=0.0, delta_v=0.002):
    return {
        "reward_terms": {"shaping": shaping, "fuel": fuel, "terminal": terminal},
        "delta_v": delta_v,
        "position": np.array([10.0, -5.0]),
        "velocity": np.array([-0.1, 0.02]),
        "thrust": np.array([0.1, 0.2]),
        "outcome": outcome,
    }


def run_episode(callback, length, outcome, terminal, n_envs=1, env=0):
    """Feed ``length`` steps to the callback, ending with ``outcome`` on ``env``."""
    for k in range(length):
        last = k == length - 1
        infos = [step_info() for _ in range(n_envs)]
        dones = np.zeros(n_envs, dtype=bool)
        if last:
            infos[env] = step_info(outcome=outcome, terminal=terminal)
            dones[env] = True
        callback.locals = {"infos": infos, "dones": dones}
        callback._on_step()


def test_rolling_mean():
    np.testing.assert_allclose(rolling_mean([1, 2, 3, 4, 5], 2), [1.0, 1.5, 2.5, 3.5, 4.5])
    np.testing.assert_allclose(rolling_mean([0, 1, 1], 10), [0.0, 0.5, 2 / 3])


def test_true_return_excludes_shaping():
    callback = LiveViewCallback(RecordingView(), episode_stride=1000)
    run_episode(callback, length=10, outcome=Outcome.DOCKED, terminal=100.0)
    # Ten steps of fuel -0.02 plus the docking bonus; the +5 of shaping per
    # step must not appear.
    assert callback.curves.true_return == [pytest.approx(100.0 - 10 * 0.02)]
    assert callback.curves.delta_v == [pytest.approx(10 * 0.002)]
    assert callback.curves.outcomes == [Outcome.DOCKED]


def test_success_rate_bookkeeping():
    callback = LiveViewCallback(RecordingView(), episode_stride=1000)
    for outcome in (Outcome.DOCKED, Outcome.ESCAPED, Outcome.DOCKED, Outcome.TIMEOUT):
        run_episode(callback, length=3, outcome=outcome, terminal=0.0)
    np.testing.assert_array_equal(callback.curves.success(), [1.0, 0.0, 1.0, 0.0])


def test_accumulators_reset_between_episodes():
    callback = LiveViewCallback(RecordingView(), episode_stride=1000)
    run_episode(callback, length=5, outcome=Outcome.TIMEOUT, terminal=0.0)
    run_episode(callback, length=2, outcome=Outcome.TIMEOUT, terminal=0.0)
    assert callback.curves.delta_v == [pytest.approx(5 * 0.002), pytest.approx(2 * 0.002)]


def test_replays_once_every_stride_and_only_env_index():
    view = RecordingView()
    callback = LiveViewCallback(view, episode_stride=3, env_index=0)
    for n in range(9):
        run_episode(callback, length=4, outcome=Outcome.TIMEOUT, terminal=0.0, n_envs=2, env=n % 2)
    # Nine episodes, alternating between env 0 and env 1. Replays can only
    # happen when env 0 finishes, once at least three episodes have passed.
    assert callback.curves.n_episodes == 9
    assert [episode for _, _, episode in view.episodes] == [3, 7]
    assert callback.replays == 2


def test_replayed_trajectory_is_the_whole_episode():
    view = RecordingView()
    callback = LiveViewCallback(view, episode_stride=1)
    run_episode(callback, length=7, outcome=Outcome.CRASHED, terminal=-100.0)
    run_episode(callback, length=4, outcome=Outcome.DOCKED, terminal=100.0)
    assert [(n, o) for n, o, _ in view.episodes] == [
        (7, Outcome.CRASHED),
        (4, Outcome.DOCKED),
    ]


def spiral_episode(n=400):
    t = np.linspace(0.0, 1.0, n)
    positions = np.column_stack([150.0 * (1 - t), 80.0 * np.sin(3 * t) * (1 - t)])
    velocities = np.gradient(positions, axis=0)
    thrusts = np.full((n, 2), 0.5)
    return positions, velocities, thrusts


def test_live_view_draws_headless(tmp_path):
    view = LiveView.from_env(RendezvousEnv())
    assert not view.interactive
    curves = TrainingCurves()
    for k in range(20):
        curves.record_episode(Outcome.DOCKED if k % 3 else Outcome.ESCAPED, -3.0 + k, 0.5)
    view.update_curves(curves)

    view.show_episode(*spiral_episode(), Outcome.DOCKED, 20)
    assert view._banner.get_text() == "DOCKED!"
    path = tmp_path / "window.png"
    view.save(str(path))
    view.close()
    assert path.stat().st_size > 10_000


def test_hud_reports_the_true_final_state():
    env = RendezvousEnv()
    view = LiveView.from_env(env)
    positions, velocities, thrusts = spiral_episode(n=150)
    positions[-1] = [3.0, 4.0]
    velocities[-1] = [0.3, -0.4]
    view.show_episode(positions, velocities, thrusts, Outcome.CRASHED, 7)
    view.close()

    hud = view.hud
    assert hud["time"] == pytest.approx(len(positions) * env.config.time_step)
    assert hud["distance"] == pytest.approx(5.0)
    assert hud["speed"] == pytest.approx(0.5)
    # Glide slope at 5 m: 0.05 + 5 / 200 = 0.075 m/s, so 0.5 m/s is too fast.
    assert hud["limit"] == pytest.approx(0.075)
    assert hud["within_limit"] is False
    # Half the thrust on both axes, |u| = 0.5 sqrt(2) u_max, for 150 steps, out
    # of a tank of sqrt(2) u_max for the whole episode: 0.5 * 150 / 300 used.
    cfg = env.config
    assert hud["fuel_left"] == pytest.approx(1.0 - 0.5 * 150 / cfg.max_episode_steps)


def test_short_ppo_run_feeds_the_window():
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    config = EnvConfig(max_episode_steps=50)
    vec_env = make_vec_env(lambda: RendezvousEnv(config), n_envs=2, seed=0)
    view = RecordingView()
    callback = LiveViewCallback(view, episode_stride=2)
    model = PPO("MlpPolicy", vec_env, n_steps=64, batch_size=64, n_epochs=1, seed=0)
    model.learn(total_timesteps=512, callback=callback)

    # 512 steps over two environments, episodes of at most 50 steps.
    assert callback.curves.n_episodes >= 512 // 50
    assert view.episodes, "no episode was replayed"
    assert view.curve_updates == 4
