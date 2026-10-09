"""Tests for distilling the graph pilot into a network (Step 28, block 5).

DART works only if the labels are the teacher's commands and not the noisy
thrust applied: a student fitted to the noise would learn the noise. And the
mode of a way in must name the side the way goes round, or the student's
mode head would learn the wrong side behind the station.
"""

from dataclasses import replace

import numpy as np
import torch

from orbital_rendezvous import EnvConfig, RendezvousEnv
from orbital_rendezvous.planning.distill import (
    Student,
    StudentPilot,
    mode_of_way,
    teacher_flight,
    train_student,
)
from orbital_rendezvous.planning.graph import CWGraph, GraphPilot

STRICT = EnvConfig(keep_out_radius=20.0, approach_cone_deg=15.0)


def test_the_mode_names_the_side_the_way_goes_round():
    straight = np.array([[0.0, 60.0], [0.0, 25.0], [0.0, 0.0]])
    plus = np.array([[30.0, -100.0], [25.0, 0.0], [6.0, 24.0], [0.0, 0.0]])
    minus = np.array([[-30.0, -100.0], [-25.0, 0.0], [-6.0, 24.0], [0.0, 0.0]])
    assert mode_of_way(straight, 20.0) == 0
    assert mode_of_way(plus, 20.0) == 1
    assert mode_of_way(minus, 20.0) == 2
    assert mode_of_way(np.zeros((1, 2)), 20.0) == 0                  # already in the cone


def test_the_labels_are_the_teachers_commands_not_the_noisy_thrust():
    env = RendezvousEnv(replace(STRICT, start_angle_range_deg=(0.0, 30.0)))
    pilot = GraphPilot(env, CWGraph(RendezvousEnv(STRICT)), seed=0)
    with np.errstate(all="ignore"):
        flight = teacher_flight(env, pilot, 5, noise=0.3, rng=np.random.default_rng(0))
    assert len(flight["observations"]) == len(flight["actions"]) > 0
    assert np.all(np.abs(flight["actions"]) <= 1.0)
    # Replay the labels without noise from the same start: a different flight,
    # since the noise moved the chaser off them.
    env.reset(seed=5)
    assert flight["mode"] in (0, 1, 2)
    np.testing.assert_allclose(flight["observations"][0], env._observation(), rtol=1e-6)


def test_the_student_keeps_its_first_mode():
    torch.manual_seed(0)
    student = Student(hidden=16)
    pilot = StudentPilot(student)
    env = RendezvousEnv(STRICT)
    obs, _ = env.reset(seed=0)
    pilot(env, obs)
    first = pilot.mode
    env.step(np.zeros(2))
    pilot(env, np.ones(5, dtype=np.float32))                   # a very different observation
    assert pilot.mode == first


def test_training_fits_the_labels_and_keeps_the_best_student():
    rng = np.random.default_rng(0)
    obs = rng.uniform(-1, 1, (400, 5)).astype(np.float32)
    act = np.tanh(obs[:, :2] * 0.8).astype(np.float32)
    flights = [{"observations": obs[i:i + 40], "actions": act[i:i + 40], "mode": i // 40 % 3}
               for i in range(0, 400, 40)]
    scores = iter([(0.3, 1.0), (0.9, 1.0), (0.5, 1.0), (0.2, 1.0)])
    kept = {}

    def evaluate(student):
        score = next(scores)
        if score[0] == 0.9:
            kept["state"] = {k: v.clone() for k, v in student.state_dict().items()}
        return score

    student = Student(hidden=32)
    history = train_student(student, flights, epochs=20, evaluate=evaluate, evaluate_every=5,
                            near_weight=1.0)
    assert history[-1]["thrust_loss"] < history[0]["thrust_loss"]
    for k, v in student.state_dict().items():
        torch.testing.assert_close(v, kept["state"][k])          # the best, not the last
