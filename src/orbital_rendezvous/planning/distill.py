"""Distilling the graph pilot into one network (Step 28, block 5).

The graph pilot of `graph` docks 198 of 200 with no way written by hand, but it
is a graph, a shortest path and an MPC: no learning. Here a network learns to
fly like it, by imitation, as the network of Step 20 learned from the V-bar
procedure's pilots (596 of 600). The difference is the teacher: nobody wrote
the way into this one, it comes out of the graph.

**Data, with DART.** The teacher flies training starts; every step records the
student's observation and the teacher's command as the label. On every flight
the thrust actually applied is the label plus Gaussian noise, while the label
stays the teacher's own (Laskey et al., 2017): the chaser drifts off the
teacher's path, the teacher, who replans, says how to come back, and the
student learns to recover from its own small errors. Imitation of fixed
trajectories cannot do this, which is why Go-Explore's phase 2 by cloning
docked 2 of 200: its trajectories could not be asked again. Near the port the
corridor leaves no room, so the noise is weaker within ``noise_radius``.

**Two heads.** Behind the station the way goes round one side or the other,
and the boundary is sharp; a single continuous network would average the two
into flying straight into the sphere. The student therefore has a mode head,
straight in or round the ``+x`` or ``-x`` side, chosen once from the start and
kept, and a thrust head that sees the mode:

    L = mean_w ||pi(o_t, m) - a_t||^2 + beta * mean(-log p(m_teacher | o_0)),

with steps within ``near_radius`` of the station weighing ``near_weight`` more:
there the cone is a few tens of centimetres wide. The student kept is the one
that docks most often on validation starts, not the last.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn

from orbital_rendezvous.core.env import RendezvousEnv
from orbital_rendezvous.core.rewards import Outcome

MODES = ("straight", "round +x", "round -x")


def mode_of_way(points: np.ndarray, keep_out: float, straight_deg: float = 30.0) -> int:
    """The mode of a way in: 0 if none of its nodes outside the sphere is further than
    ``straight_deg`` from the axis, otherwise 1 or 2 for the side, ``+x`` or ``-x``, of
    the node furthest round.

    Not the node nearest the station: that is often the mouth of the cone, on
    the axis, and every way round would have read as straight in.
    """
    outside = points[np.hypot(points[:, 0], points[:, 1]) > keep_out]
    if len(outside) == 0:
        return 0
    angles = np.degrees(np.arctan2(np.abs(outside[:, 0]), outside[:, 1]))
    furthest = int(np.argmax(angles))
    if angles[furthest] <= straight_deg:
        return 0
    return 1 if outside[furthest, 0] > 0 else 2


def teacher_flight(
    env: RendezvousEnv,
    pilot,
    seed: int,
    noise: float,
    rng: np.random.Generator,
    noise_near: float = 0.02,
    noise_radius: float = 40.0,
) -> dict[str, Any]:
    """One approach flown by the graph pilot from the start ``seed`` selects, with DART noise.

    Returns the student's observations, the teacher's commands as labels, the
    mode of the teacher's first way in, and whether it docked. The thrust
    applied is the label plus Gaussian noise of standard deviation ``noise``,
    ``noise_near`` within ``noise_radius`` of the station.
    """
    obs, _ = env.reset(seed=seed)
    observations, actions, done, mode = [], [], False, 0
    while not done:
        action = np.clip(np.asarray(pilot(env, obs), dtype=float), -1.0, 1.0)
        if env.steps == 0:
            ways = pilot.graph.points[pilot.way]
            mode = mode_of_way(ways, env.config.keep_out_radius)
        observations.append(obs)
        actions.append(action)
        sigma = noise if float(np.hypot(*env.state[:2])) > noise_radius else noise_near
        applied = np.clip(action + sigma * rng.standard_normal(2), -1.0, 1.0)
        obs, _, terminated, truncated, info = env.step(applied)
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
    """The student as a controller: its mode chosen at the first step and kept."""

    def __init__(self, student: Student) -> None:
        self.student = student.eval()
        self.mode = 0

    def __call__(self, env, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32))[None]
        with torch.no_grad():
            if env.steps == 0:
                self.mode = int(self.student.mode_logits(x).argmax())
            return self.student.thrust(x, torch.tensor([self.mode]))[0].numpy()


def train_student(
    student: Student,
    flights: list[dict[str, Any]],
    epochs: int = 40,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    final_learning_rate: float = 1e-5,
    beta: float = 1.0,
    near_weight: float = 10.0,
    near_radius: float = 20.0,
    position_scale: float = 500.0,
    evaluate: Callable[[Student], tuple[float, float]] | None = None,
    evaluate_every: int = 5,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Fit the student to the teacher's flights, the rate decaying along a cosine;
    keep the student that docks most often in closed loop, cheaper on a tie."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    obs = torch.as_tensor(np.concatenate([f["observations"] for f in flights]))
    act = torch.as_tensor(np.concatenate([f["actions"] for f in flights]))
    step_mode = torch.as_tensor(np.concatenate(
        [np.full(len(f["actions"]), f["mode"]) for f in flights]), dtype=torch.long)
    near = torch.linalg.norm(obs[:, :2] * position_scale, dim=1) < near_radius
    weights = 1.0 + (near_weight - 1.0) * near.float()
    start_obs = torch.as_tensor(np.stack([f["observations"][0] for f in flights]))
    start_mode = torch.as_tensor([f["mode"] for f in flights], dtype=torch.long)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                          eta_min=final_learning_rate)
    history, best, best_state = [], None, None
    for epoch in range(epochs):
        student.train()
        order = rng.permutation(len(obs))
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            starts = rng.integers(0, len(start_obs), size=min(256, len(start_obs)))
            error = ((student.thrust(obs[batch], step_mode[batch]) - act[batch]) ** 2).sum(dim=1)
            thrust_loss = (weights[batch] * error).sum() / weights[batch].sum()
            mode_loss = nn.functional.cross_entropy(student.mode_logits(start_obs[starts]),
                                                    start_mode[starts])
            loss = thrust_loss + beta * mode_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        schedule.step()
        row = {"epoch": epoch, "thrust_loss": float(thrust_loss.detach()),
               "mode_loss": float(mode_loss.detach())}
        if evaluate is not None and (epoch + 1) % evaluate_every == 0:
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
    torch.save(student.state_dict(), path)


def load_student(path, hidden: int = 256) -> Student:
    student = Student(hidden=hidden)
    student.load_state_dict(torch.load(path, map_location="cpu"))
    return student.eval()
