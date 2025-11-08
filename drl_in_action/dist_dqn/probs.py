import numpy as np


def get_target_probs(
    probs_batch: np.ndarray,
    action_batch: np.ndarray,
    reward_batch: np.ndarray,
    done_batch: np.ndarray,
    lims: tuple[float, float],
    gamma: float,
) -> np.ndarray:
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
    v_min, v_max = lims
    delta_z = (v_max - v_min) / (n - 1.0)
    new_probs = probs_batch.copy()
    for i in range(probs_batch.shape[0]):
        action = action_batch[i]
        probs = probs_batch[i, action]
        if done_batch[i]:
            target_probs = np.zeros(n)
            bj = int(np.clip(np.round((reward_batch[i] - v_min) / delta_z), 0, n - 1))
            target_probs[bj] = 1.0
        else:
            target_probs = update_probs(probs, reward_batch[i], lims, gamma)
        new_probs[i, action, :] = target_probs
    return new_probs


def update_probs(
    probs: np.ndarray,
    r: float,
    limit: tuple[float, float],
    gamma: float,
) -> np.ndarray:
    """
    args:
        probs: a 1D array of probabilities
        r: the rewards
        limit: support limit min and max
        gamma: the discount factor from 0 to 1
    returns:
        probs: the new reward probabilities
    """
    n = probs.shape[0]
    reward_min, reward_max = limit
    delta_z = (reward_max - reward_min) / (n - 1)
    reward_idx = int(np.clip(np.round((r - reward_min) / delta_z), 0, n - 1))
    new_probs = probs.copy()
    j = 1
    for i in range(reward_idx, 1, -1):
        new_probs[i] += np.power(gamma, j) * new_probs[i - 1]
        j += 1
    j = 1
    for i in range(reward_idx, n - 1, 1):
        new_probs[i] += np.pow(gamma, j) * new_probs[i + 1]
        j += 1
    new_probs /= new_probs.sum()
    return new_probs
