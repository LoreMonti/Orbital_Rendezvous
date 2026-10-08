"""Tests for keeping the best policy seen during training, not the last one.

PPO can learn a manoeuvre and lose it later; the callback must save a policy
only when it docks more often, or as often for less, and can count time in
that cost too.
"""

from types import SimpleNamespace

import pytest

from orbital_rendezvous import Outcome
from orbital_rendezvous.rl.callbacks import BestModel


def test_best_model_can_count_time_in_its_tie_break(tmp_path):
    best = BestModel(lambda: None, tmp_path / "best.zip", time_weight=2e-4)
    runs = [SimpleNamespace(outcome=Outcome.DOCKED, delta_v=0.9, time=2000.0),
            SimpleNamespace(outcome=Outcome.DOCKED, delta_v=1.0, time=1000.0)]
    import orbital_rendezvous.core.evaluation as evaluation
    original = evaluation.evaluate
    evaluation.evaluate = lambda env, controller, seeds: runs
    try:
        best._env, best.model = object(), SimpleNamespace(predict=None)
        success, cost = best.measure()
    finally:
        evaluation.evaluate = original
    # 0.9 + 0.4 and 1.0 + 0.2: the median of 1.3 and 1.2.
    assert success == 1.0 and cost == pytest.approx(1.25)


def test_best_model_keeps_the_best_not_the_last(tmp_path):
    best = BestModel(lambda: None, tmp_path / "best.zip", evaluate_every=1)
    saved = []
    best.model = SimpleNamespace(save=lambda path: saved.append(best.num_timesteps),
                                 logger=SimpleNamespace(record=lambda k, v: None))
    readings = iter([(0.6, 1.0), (0.9, 1.2), (0.9, 1.1), (0.4, 0.8), (0.9, 1.3)])
    best.measure = lambda: next(readings)
    for step in range(5):
        best.num_timesteps = step
        best._on_rollout_end()
    # Saved when docking improved, and on a tie only with less fuel.
    assert saved == [0, 1, 2]
    assert best.best == (0.9, 1.1)
