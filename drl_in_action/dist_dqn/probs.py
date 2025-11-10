import numpy as np
import torch


def get_target_probs(
    probs_batch: torch.Tensor,
    action_batch: torch.Tensor,
    reward_batch: torch.Tensor,
    done_batch: torch.Tensor,
    lims: tuple[float, float],
    gamma: float,
) -> torch.Tensor:
    """
    args:
        probs_batch: [BATCH, NUM_ACTIONS, NUM_SUPPORT]
        action_batch: [BATCH]
        reward_batch: [BATCH]
        done_batch: [BATCH]
        lims: (v_min, v_max)
    returns:
        new probs batch [BATCH, NUM_ACTIONS, NUM_SUPPORT]
    """
    n = probs_batch.shape[-1]
    support_indices = torch.arange(n).to(probs_batch.device)
    v_min, v_max = lims
    delta_z = (v_max - v_min) / (n - 1.0)
    new_probs = probs_batch.clone()
    for i in range(probs_batch.shape[0]):
        action = action_batch[i]
        probs = probs_batch[i, action]
        if done_batch[i]:
            target_probs = torch.zeros(n).to(probs_batch.device)
            bj = torch.clip(
                torch.round((reward_batch[i] - v_min) / delta_z), 0, n - 1
            ).long()
            target_probs[bj] = 1.0
        else:
            target_probs = update_probs(probs, reward_batch[i], lims, gamma)
        new_probs[i, action, support_indices] = target_probs
    return new_probs


def update_probs(
    probs: torch.Tensor,
    r: torch.Tensor,
    limit: tuple[float, float],
    gamma: float,
) -> torch.Tensor:
    """
    args:
        probs: a 1D array of probabilities
        r: the reward (single number)
        limit: support limit min and max
        gamma: the discount factor from 0 to 1
    returns:
        probs: the new reward probabilities
    """
    n = probs.shape[0]
    reward_min, reward_max = limit
    delta_z = (reward_max - reward_min) / (n - 1)
    reward_idx = int(torch.clip(torch.round((r - reward_min) / delta_z), 0, n - 1))
    new_probs = probs.clone()
    j = torch.tensor(1).to(probs.device)
    for i in range(reward_idx, 1, -1):
        new_probs[i] += torch.pow(gamma, j) * new_probs[i - 1]
        j += 1
    j = torch.tensor(1).to(probs.device)
    for i in range(reward_idx, n - 1, 1):
        new_probs[i] += torch.pow(gamma, j) * new_probs[i + 1]
        j += 1
    new_probs /= new_probs.sum()
    return new_probs
