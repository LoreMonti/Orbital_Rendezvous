"""Tests for distilling the planner and the pilots into one network (Step 20).

The labels must be the teacher's own commands, whatever noise is applied to
the flight; noise must stay out of the region near the station where it is
switched off. The student must be able to learn a choice that jumps with its
mode head and keep the mode it chose for the whole approach, and training
must keep the student that flies best, not the last one.
"""

import numpy as np
import torch

from orbital_rendezvous import EnvConfig, RendezvousEnv
from teacher_student.distillation import (
    Student,
    StudentPilot,
    load_flights,
    load_student,
    mode_of,
    save_flights,
    save_student,
    step_weights,
    teacher_flight,
    train_student,
)
from teacher_student.goto import waypoint_menu
from teacher_student.hierarchy import HierarchicalPilot, PlannerEnv


def test_the_mode_is_the_side_of_the_waypoint():
    assert mode_of(None) == 0
    assert mode_of(np.array([0.0, 60.0])) == 0            # on the docking axis
    assert mode_of(np.array([40.0, 0.0])) == 1
    assert mode_of(np.array([-37.0, -15.0])) == 2


def pushing_teacher(max_steps=6):
    """A teacher that always commands a fixed thrust, and a waypoint on the -x side."""
    pilot = HierarchicalPilot(lambda obs: np.array([0.3, -0.2]), lambda obs: np.zeros(2),
                              EnvConfig(docking_radius=2.0, docking_speed=0.04),
                              EnvConfig(max_episode_steps=150),
                              planner=lambda state: np.array([-40.0, 0.0]))
    corridor = RendezvousEnv(EnvConfig(keep_out_radius=20.0, max_episode_steps=max_steps))
    return PlannerEnv(corridor, pilot, waypoint_menu((40.0, 60.0), 16)), pilot


def test_labels_are_the_teachers_commands_whatever_the_noise():
    env, pilot = pushing_teacher()
    clean = teacher_flight(env, pilot, 3, 0.0, np.random.default_rng(0))
    noisy = teacher_flight(env, pilot, 3, 0.5, np.random.default_rng(0))
    assert clean["mode"] == noisy["mode"] == 2
    np.testing.assert_allclose(noisy["actions"], np.tile([0.3, -0.2], (6, 1)), rtol=1e-6)
    np.testing.assert_array_equal(clean["observations"][0], noisy["observations"][0])
    assert not np.allclose(clean["observations"][-1], noisy["observations"][-1])


def test_no_noise_is_added_near_the_station():
    env, pilot = pushing_teacher()
    clean = teacher_flight(env, pilot, 3, 0.0, np.random.default_rng(0))
    shielded = teacher_flight(env, pilot, 3, 0.5, np.random.default_rng(0), noise_radius=1e6)
    np.testing.assert_array_equal(clean["observations"], shielded["observations"])


def test_the_student_outputs_a_mode_and_a_bounded_thrust():
    student = Student()
    obs = torch.randn(7, 5)
    assert student.mode_logits(obs).shape == (7, 3)
    thrust = student.thrust(obs, torch.zeros(7, dtype=torch.long))
    assert thrust.shape == (7, 2) and thrust.abs().max() <= 1.0


def test_the_student_keeps_the_mode_it_chose_at_the_start():
    student = Student()
    with torch.no_grad():
        # Mode scores that follow the sign of the first observation.
        student.mode_head[-1].weight.zero_()
        student.mode_head[-1].bias.copy_(torch.tensor([0.0, 1.0, -1.0]))
    pilot = StudentPilot(student)
    env = type("Env", (), {"steps": 0})()
    pilot(env, np.zeros(5))
    assert pilot.mode == 1
    with torch.no_grad():
        student.mode_head[-1].bias.copy_(torch.tensor([0.0, -1.0, 1.0]))
    env.steps = 4
    pilot(env, np.zeros(5))
    assert pilot.mode == 1                                  # decided once, and kept


def flights_with_a_jump(n=400, length=10, seed=0):
    """Mode 1 and thrust +0.8 for starts with x < 0, mode 2 and thrust -0.8 otherwise."""
    rng = np.random.default_rng(seed)
    flights = []
    for _ in range(n):
        obs = rng.uniform(-1.0, 1.0, size=(length, 5)).astype(np.float32)
        mode = 1 if obs[0, 0] < 0 else 2
        action = np.full((length, 2), 0.8 if mode == 1 else -0.8, dtype=np.float32)
        flights.append({"observations": obs, "actions": action, "mode": mode})
    return flights


def test_training_learns_a_choice_that_jumps_and_the_thrust_for_it():
    student = Student(hidden=64)
    train_student(student, flights_with_a_jump(), epochs=30, batch_size=256, seed=0)
    starts = torch.zeros(2, 5)
    starts[:, 0] = torch.tensor([-0.2, 0.2])
    np.testing.assert_array_equal(student.mode_logits(starts).argmax(1).numpy(), [1, 2])
    thrust = student.thrust(starts, torch.tensor([1, 2])).detach().numpy()
    assert thrust[0, 0] > 0.6 and thrust[1, 0] < -0.6


def test_training_keeps_the_student_that_flew_best():
    student = Student(hidden=32)
    scores = iter([(0.5, 70.0), (0.9, 66.0), (0.9, 64.0), (0.6, 60.0)])
    snapshots = []

    def evaluate(s):
        snapshots.append({k: v.clone() for k, v in s.state_dict().items()})
        return next(scores)

    history = train_student(student, flights_with_a_jump(50, 4), epochs=4, batch_size=64,
                            evaluate=evaluate, evaluate_every=1)
    assert [h["success"] for h in history] == [0.5, 0.9, 0.9, 0.6]
    for key, value in student.state_dict().items():
        torch.testing.assert_close(value, snapshots[2][key])   # 0.9 docked, and cheaper


def test_a_saved_student_flies_the_same(tmp_path):
    student = Student(hidden=32)
    save_student(student, tmp_path / "student.pt")
    loaded = load_student(tmp_path / "student.pt")
    obs = torch.randn(3, 5)
    torch.testing.assert_close(loaded.thrust(obs, torch.tensor([0, 1, 2])),
                               student.thrust(obs, torch.tensor([0, 1, 2])))
    assert loaded.trunk[0].out_features == 32
    assert not loaded.training


def test_steps_near_the_station_weigh_more():
    # Positions are observed divided by 500 m: 10 m, 19.9 m, 20.1 m and 150 m out.
    obs = torch.zeros(4, 5)
    obs[:, 1] = torch.tensor([10.0, 19.9, 20.1, 150.0]) / 500.0
    np.testing.assert_allclose(step_weights(obs, 10.0, 20.0, 500.0).numpy(), [10, 10, 1, 1])
    np.testing.assert_allclose(step_weights(obs, 1.0, 20.0, 500.0).numpy(), [1, 1, 1, 1])


def test_saved_flights_come_back_as_views_of_one_array(tmp_path):
    flights = [dict(f, docked=True, noise=0.1) for f in flights_with_a_jump(5, 3)]
    flights[2]["observations"] = flights[2]["observations"][:2]
    flights[2]["actions"] = flights[2]["actions"][:2]
    save_flights(flights, tmp_path / "flights.npz")
    loaded = load_flights(tmp_path / "flights.npz")
    assert [len(f["actions"]) for f in loaded] == [3, 3, 2, 3, 3]
    for original, back in zip(flights, loaded, strict=True):
        np.testing.assert_array_equal(back["observations"], original["observations"])
        assert back["mode"] == original["mode"] and back["noise"] == 0.1
    # One array in memory, not one per flight: every flight shares its base.
    assert all(f["observations"].base is loaded[0]["observations"].base for f in loaded)
