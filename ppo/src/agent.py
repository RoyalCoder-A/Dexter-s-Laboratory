from re import S

import numpy as np
import torch
from ppo.src.model_implementation import PPOModel


class Agent:
    def __init__(
        self, model: PPOModel, clip_range: float, gae_lambda: float, gamma: float
    ):
        self.model = model
        self.clip_range = clip_range
        self.gae_lambda = gae_lambda
        self.gamma = gamma

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

    def _calculate_advantage(
        self,
        values: torch.Tensor,
        next_values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ):
        advantages = torch.zeros_like(rewards)
        gae = 0

        # We need to go backwards through time
        for t in reversed(range(len(rewards))):
            # Use the provided next_values directly
            next_value = next_values[t] * (1 - dones[t])

            # Calculate delta (TD error) according to equation 12
            delta = rewards[t] + self.gamma * next_value - values[t]

            # Calculate GAE according to equation 11 (recursive form)
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        return advantages

    def _calculate_value_targets(
        self, rewards: torch.Tensor, next_values: torch.Tensor, dones: torch.Tensor
    ):
        value_targets = torch.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            # For each timestep, the value target is just the reward plus discounted next value
            value_targets[t] = rewards[t] + self.gamma * next_values[t] * (1 - dones[t])
        return value_targets
