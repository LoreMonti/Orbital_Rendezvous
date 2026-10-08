"""Distillation: one network that flies what the planner and the two pilots fly together.

The teacher is the system of Step 17c: a learned planner choosing the waypoint,
a go-to pilot flying to it and to the hold point, a final-approach pilot down
the corridor, glued by a menu, thresholds and a fixed sequence. The student is
a single network with the observation and the action of the default agent,
five numbers in and two thrusts out, and nothing hand-written around it.

**Data.** The teacher flies training starts; every step records what the
student would see and the thrust the teacher commands. On a share of the
flights the thrust actually applied is the teacher's plus Gaussian noise,
while the label stays the teacher's own (DART, Laskey et al., 2017): the
chaser drifts off the teacher's path, and the data show how to come back,
which a student that errs a little needs. Near the port the corridor leaves
no room: with noise of 0.05 of full thrust everywhere the teacher itself
failed one flight in four, and the data would lack the last metres. Noise can
therefore be limited to beyond ``noise_radius`` from the station.

**Two heads.** Behind the station the teacher goes around on one side or the
other, and the boundary between the two is sharp; a continuous network
fitted to both would average them and fly straight into the station. The
student therefore has a mode head, a choice among three, straight to the hold
point or around on the ``+x`` or ``-x`` side, taken once from the start and
kept, and a thrust head that sees the mode chosen:

    L = mean ||pi(o_t, m) - a_t||^2 + beta * mean(-log p(m_teacher | s_0)).

The three scores are smooth, but the choice, their argmax, can jump; given
the mode, the thrust never has to.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn

from orbital_rendezvous.core.rewards import Outcome

from .hierarchy import HierarchicalPilot, PlannerEnv

MODES = ("straight", "around +x", "around -x")


def mode_of(waypoint: np.ndarray | None) -> int:
    """0 without a waypoint or with one on the docking axis, 1 on the +x side, 2 on the -x side."""
    if waypoint is None or abs(float(waypoint[0])) < 1e-6:
        return 0
    return 1 if waypoint[0] > 0 else 2


def learned_planner(model, env: PlannerEnv) -> Callable[[np.ndarray], np.ndarray | None]:
    """The planner of Step 17c as a function of the start state: its waypoint, or None."""
    def plan(state: np.ndarray) -> np.ndarray | None:
        saved = env.corridor.state
        env.corridor.state = state
        obs = env._observation()
        env.corridor.state = saved
        return env.waypoint(int(model.predict(obs, deterministic=True)[0]))
    return plan


def teacher_flight(
    env: PlannerEnv,
    pilot: HierarchicalPilot,
    seed: int,
    noise: float,
    rng: np.random.Generator,
    noise_radius: float = 0.0,
) -> dict[str, Any]:
    """One approach flown by the teacher from the start ``seed`` selects.

    Returns the student's observations, the teacher's thrust commands as
    labels, the mode of the teacher's plan, and how the flight ended. With
    ``noise``, the thrust applied is the label plus Gaussian noise of that
    standard deviation, clipped to the thruster's range, while the chaser is
    further than ``noise_radius`` from the station.
    """
    corridor = env.corridor
    obs, _ = corridor.reset(seed=seed)
    observations, actions, done = [], [], False
    while not done:
        action = np.clip(np.asarray(pilot(corridor, obs), dtype=float), -1.0, 1.0)
        if corridor.steps == 0:
            mode = mode_of(pilot.plan[0] if len(pilot.plan) > 1 else None)
        observations.append(obs)
        actions.append(action)
        noisy = noise and float(np.hypot(*corridor.state[:2])) > noise_radius
        applied = np.clip(action + noise * rng.standard_normal(2), -1.0, 1.0) if noisy else action
        obs, _, terminated, truncated, info = corridor.step(applied)
        done = terminated or truncated
    return {"observations": np.array(observations, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32), "mode": mode,
            "docked": info["outcome"] is Outcome.DOCKED}


class Student(nn.Module):
    """The single network: a shared trunk, a mode head and a thrust head that sees the mode."""

    def __init__(self, obs_dim: int = 5, hidden: int = 256, modes: int = len(MODES)) -> None:
        super().__init__()
        self.modes = modes
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())
        self.mode_head = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(), nn.Linear(64, modes))
        self.thrust_head = nn.Sequential(nn.Linear(hidden + modes, hidden), nn.Tanh(),
                                         nn.Linear(hidden, 2), nn.Tanh())

    def mode_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mode_head(self.trunk(obs))

    def thrust(self, obs: torch.Tensor, mode: torch.Tensor) -> torch.Tensor:
        one_hot = nn.functional.one_hot(mode, self.modes).float()
        return self.thrust_head(torch.cat([self.trunk(obs), one_hot], dim=-1))


class StudentPilot:
    """The student as a controller, ``(env, obs) -> action``: it picks its mode at the first
    step, from its own mode head, and keeps it for the whole approach."""

    def __init__(self, student: Student) -> None:
        self.student = student.eval()
        self.mode = 0

    def __call__(self, env, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32))[None]
        with torch.no_grad():
            if env.steps == 0:
                self.mode = int(self.student.mode_logits(x).argmax())
            return self.student.thrust(x, torch.tensor([self.mode]))[0].numpy()


def step_weights(
    obs: torch.Tensor, near_weight: float, near_radius: float, position_scale: float
) -> torch.Tensor:
    """``1 + (near_weight - 1) [r < near_radius]`` per step, ``r`` read from its observation."""
    near = torch.linalg.norm(obs[:, :2] * position_scale, dim=1) < near_radius
    return 1.0 + (near_weight - 1.0) * near.float()


def train_student(
    student: Student,
    flights: list[dict[str, Any]],
    epochs: int = 40,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    beta: float = 1.0,
    evaluate: Callable[[Student], tuple[float, float]] | None = None,
    evaluate_every: int = 5,
    evaluate_all_from: int | None = None,
    near_weight: float = 1.0,
    near_radius: float = 20.0,
    position_scale: float = 500.0,
    final_learning_rate: float | None = None,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Fit the student to the teacher's flights; keep the best student in closed loop.

    Each step of Adam takes a batch of steps for the thrust loss and a batch of
    flights for the mode loss, ``beta`` weighing the second. Every
    ``evaluate_every`` epochs ``evaluate`` flies the student and returns its
    docking rate and median cost; the student that docks most often, cheaper
    on a tie, is kept, since what matters is how it flies, not how close it
    copies; from epoch ``evaluate_all_from`` on, every epoch is flown.

    Steps within ``near_radius`` of the station weigh ``near_weight`` times
    more in the thrust loss: there the approach cone is a few tens of
    centimetres wide, and an error that is harmless 150 m out makes the chaser
    leave it. Observations hold positions divided by ``position_scale``. With
    ``final_learning_rate``, the rate decays along a cosine to it.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    obs = torch.as_tensor(np.concatenate([f["observations"] for f in flights]))
    act = torch.as_tensor(np.concatenate([f["actions"] for f in flights]))
    step_mode = torch.as_tensor(np.concatenate(
        [np.full(len(f["actions"]), f["mode"]) for f in flights]), dtype=torch.long)
    weights = step_weights(obs, near_weight, near_radius, position_scale)
    start_obs = torch.as_tensor(np.stack([f["observations"][0] for f in flights]))
    start_mode = torch.as_tensor([f["mode"] for f in flights], dtype=torch.long)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    schedule = None if final_learning_rate is None else torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=final_learning_rate)
    history, best, best_state = [], None, None
    for epoch in range(epochs):
        student.train()
        order = rng.permutation(len(obs))
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            flights_batch = rng.integers(0, len(start_obs), size=min(256, len(start_obs)))
            error = ((student.thrust(obs[batch], step_mode[batch]) - act[batch]) ** 2).sum(dim=1)
            thrust_loss = (weights[batch] * error).sum() / weights[batch].sum()
            mode_loss = nn.functional.cross_entropy(
                student.mode_logits(start_obs[flights_batch]), start_mode[flights_batch])
            loss = thrust_loss + beta * mode_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if schedule is not None:
            schedule.step()
        row = {"epoch": epoch, "thrust_loss": float(thrust_loss.detach()),
               "mode_loss": float(mode_loss.detach())}
        late = evaluate_all_from is not None and epoch >= evaluate_all_from
        if evaluate is not None and ((epoch + 1) % evaluate_every == 0 or late):
            student.eval()
            score = evaluate(student)
            row.update(success=score[0], cost=score[1])
            if best is None or score[0] > best[0] or (score[0] == best[0] and score[1] < best[1]):
                best = score
                best_state = {k: v.clone() for k, v in student.state_dict().items()}
        history.append(row)
    if best_state is not None:
        student.load_state_dict(best_state)
    return history


def save_student(student: Student, path) -> None:
    torch.save({"state_dict": student.state_dict(),
                "hidden": student.trunk[0].out_features,
                "obs_dim": student.trunk[0].in_features}, path)


def load_student(path) -> Student:
    saved = torch.load(path, weights_only=False)
    student = Student(saved["obs_dim"], saved["hidden"])
    student.load_state_dict(saved["state_dict"])
    return student.eval()


def save_flights(flights: list[dict[str, Any]], path) -> None:
    """The teacher's flights in one file: steps concatenated, with the length of each flight."""
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, observations=np.concatenate([f["observations"] for f in flights]),
             actions=np.concatenate([f["actions"] for f in flights]),
             lengths=np.array([len(f["actions"]) for f in flights]),
             modes=np.array([f["mode"] for f in flights]),
             docked=np.array([f["docked"] for f in flights]),
             noise=np.array([f.get("noise", 0.0) for f in flights]))


def load_flights(path) -> list[dict[str, Any]]:
    """The saved flights, each array read from the file once.

    Indexing an ``NpzFile`` reads and decompresses the whole array again on
    every access; done once per flight, with each slice keeping its own copy
    alive, 12 000 flights asked for hundreds of gigabytes and froze the
    computer. The arrays are read once here, and the flights are views into
    them.
    """
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    cuts = np.cumsum(arrays["lengths"])[:-1]
    observations = np.split(arrays["observations"], cuts)
    actions = np.split(arrays["actions"], cuts)
    return [{"observations": o, "actions": a, "mode": int(m), "docked": bool(d), "noise": float(s)}
            for o, a, m, d, s in zip(observations, actions, arrays["modes"], arrays["docked"],
                                     arrays["noise"], strict=True)]
