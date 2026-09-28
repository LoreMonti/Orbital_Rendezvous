"""Learning the planner's choice from the oracle, by supervised learning.

A bandit learns about one choice per approach, the one it tried; with PPO the
learned planner committed early to a few safe choices and stopped trying the
others (Step 17b). In simulation every choice can be flown from every start,
so the planner can instead be taught the whole ranking: for each training
start, `hierarchy.oracle_costs` gives the cost ``J_k`` of every choice, and the
planner's categorical policy is fitted to soft targets

    p_k = exp(-(J_k - J_min) / T) / sum_j exp(-(J_j - J_min) / T),

with ``p_k = 0`` for a choice that did not dock. Near-ties share the target
instead of one being picked by noise, and the temperature ``T``, in units of
cost, sets how near is near. The network is the same as the PPO planner's, so
the result is saved and evaluated as one.

The cheapest choice that docks is often the one that passes closest to the
keep-out sphere, so the targets alone teach the planner to cut corners, and a
small error near that edge is a violation. A second term charges the expected
cost of the policy's choice, with a failure priced as in the bandit's reward,

    L = - sum_k p_k log pi_k + lambda * sum_k pi_k (J_k + C_fail [k fails]) / C_fail,

so that probability placed on a choice that fails costs far more than a
little extra fuel saves: near the edge the planner learns to keep a margin.
"""

from __future__ import annotations

import numpy as np
import torch


def soft_targets(costs: np.ndarray, docked: np.ndarray, temperature: float) -> np.ndarray:
    """Targets over the choices of each start: a softmax of the cost saved, failures excluded.

    Starts from which no choice docked carry no information about the ranking
    and must be removed beforehand.
    """
    if not docked.any(axis=1).all():
        raise ValueError("every start needs at least one choice that docks")
    allowed = np.where(docked, costs, np.inf)
    logits = -(allowed - allowed.min(axis=1, keepdims=True)) / temperature
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def expected_costs(costs: np.ndarray, docked: np.ndarray, failure_cost: float) -> np.ndarray:
    """Each choice's cost, a failure priced in: ``(J_k + C_fail [k fails]) / C_fail``."""
    return (costs + failure_cost * ~docked) / failure_cost


def imitate(
    model,
    observations: np.ndarray,
    targets: np.ndarray,
    penalties: np.ndarray | None = None,
    risk_weight: float = 0.0,
    epochs: int = 300,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Fit the policy of a Stable-Baselines3 model to target distributions over its actions.

    Minimises the cross-entropy ``-sum_k p_k log pi(k | s)``, plus
    ``risk_weight`` times the expected penalty ``sum_k pi(k | s) penalties_k``
    if penalties are given, by Adam; keeps the parameters with the lowest loss
    on a held-out fraction of the starts, and returns the loss per epoch.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    order = rng.permutation(len(observations))
    n_val = max(1, int(validation_fraction * len(observations)))
    val, train = order[:n_val], order[n_val:]
    policy = model.policy
    obs = torch.as_tensor(observations, dtype=torch.float32)
    tgt = torch.as_tensor(targets, dtype=torch.float32)
    pen = None if penalties is None else torch.as_tensor(penalties, dtype=torch.float32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    def loss_of(index) -> torch.Tensor:
        log_probs = policy.get_distribution(obs[index]).distribution.logits
        loss = -(tgt[index] * log_probs).sum(dim=1).mean()
        if pen is not None and risk_weight:
            loss = loss + risk_weight * (log_probs.exp() * pen[index]).sum(dim=1).mean()
        return loss

    best, best_state, history = np.inf, None, []
    for epoch in range(epochs):
        policy.train()
        for start in range(0, len(train), batch_size):
            batch = train[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_of(batch)
            loss.backward()
            optimizer.step()
        rng.shuffle(train)
        policy.eval()
        with torch.no_grad():
            train_loss, val_loss = float(loss_of(train)), float(loss_of(val))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        if val_loss < best:
            best = val_loss
            best_state = {k: v.clone() for k, v in policy.state_dict().items()}
    policy.load_state_dict(best_state)
    return history
