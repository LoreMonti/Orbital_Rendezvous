"""Tests for the training window and the callback that feeds it.

The callback is driven by hand with episodes whose outcome, return and delta-v
are known in advance, so its bookkeeping is checked exactly. A short real PPO
run then checks that the pieces fit together inside Stable-Baselines3.
"""

from types import SimpleNamespace

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

    def show_episode(self, positions, thrusts, outcome, episode, delta_v):
        self.episodes.append((len(positions), outcome, episode, delta_v))

    def update_curves(self, curves):
        self.curve_updates += 1


def step_info(outcome=None, shaping=5.0, fuel=-0.02, terminal=0.0, delta_v=0.002):
    return {
        "reward_terms": {"shaping": shaping, "fuel": fuel, "terminal": terminal},
        "delta_v": delta_v,
        "position": np.array([10.0, -5.0]),
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
    assert [episode for _, _, episode, _ in view.episodes] == [3, 7]
    assert callback.replays == 2


def test_replayed_trajectory_is_the_whole_episode():
    view = RecordingView()
    callback = LiveViewCallback(view, episode_stride=1)
    run_episode(callback, length=7, outcome=Outcome.CRASHED, terminal=-100.0)
    run_episode(callback, length=4, outcome=Outcome.DOCKED, terminal=100.0)
    assert [(n, o) for n, o, _, _ in view.episodes] == [
        (7, Outcome.CRASHED),
        (4, Outcome.DOCKED),
    ]


def test_losses_are_read_from_the_logger():
    callback = LiveViewCallback(RecordingView())
    logger = SimpleNamespace(name_to_value={})
    callback.model = SimpleNamespace(logger=logger)
    callback._on_rollout_start()
    assert callback.curves.policy_loss == []
    logger.name_to_value.update({"train/policy_gradient_loss": -0.01, "train/value_loss": 3.0})
    callback._on_rollout_start()
    assert callback.curves.policy_loss == [-0.01]
    assert callback.curves.value_loss == [3.0]


def test_live_view_draws_headless(tmp_path):
    view = LiveView.from_env(RendezvousEnv())
    assert not view.interactive
    curves = TrainingCurves()
    for k in range(20):
        curves.record_episode(Outcome.DOCKED if k % 3 else Outcome.ESCAPED, -3.0 + k, 0.5)
    curves.policy_loss.extend([0.01, -0.02])
    curves.value_loss.extend([4.0, 2.0])
    view.update_curves(curves)

    t = np.linspace(0.0, 1.0, 400)
    positions = np.column_stack([150.0 * (1 - t), 80.0 * np.sin(3 * t) * (1 - t)])
    view.show_episode(positions, np.ones((400, 2)), Outcome.DOCKED, 20, 0.61)
    path = tmp_path / "window.png"
    view.save(str(path))
    view.close()
    assert path.stat().st_size > 10_000


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
    assert len(callback.curves.policy_loss) == 3
