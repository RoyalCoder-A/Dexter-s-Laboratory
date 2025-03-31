from re import S

import numpy as np
import torch
from ppo.src.model_implementation import PPOModel


class Agent:
    def __init__(self, model: PPOModel, clip_range: float):
        self.model = model
        self.clip_range = clip_range

    def sample_action(self, obs: np.array):
        """
        obs: (n_envs, obs_dim)
        """
        values, probs = self.model(
            obs
        )  # values: (n_envs, 1), probs: (n_envs, n_action)
        values = torch.squeeze(values)  # (n_envs, )
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()  # (n_envs, )
        prob_log = dist.log_prob(action)  # (n_envs, )
        return action, prob_log, values

    def _clipped_surrogate_objective(
        self,
        log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantage: torch.Tensor,
    ) -> torch.Tensor:
        """
        log_prob: (batch_size,)
        old_log_prob: (batch_size,)
        advantage: (batch_size,)
        """
        ratio = torch.exp(log_prob - old_log_prob)
        surrogate_1 = ratio * advantage
        surrogate_2 = (
            torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage
        )
        return torch.min(surrogate_1, surrogate_2)  # (batch_size,)
