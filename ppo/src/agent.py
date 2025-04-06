from re import S

import numpy as np
import torch
from ppo.src.model_implementation import PPOModel


class Agent:
    def __init__(
        self,
        model: PPOModel,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        clip_range: float,
        gae_lambda: float,
        gamma: float,
        value_coef: float,
        entropy_coef: float,
    ):
        self.model = model
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def sample_action(self, obs: np.ndarray):
        """
        obs: (n_envs, obs_dim)
        """
        self.model.eval()
        values, probs = self.model(
            obs
        )  # values: (n_envs, 1), probs: (n_envs, n_action)
        values = torch.squeeze(values)  # (n_envs, )
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()  # (n_envs, )
        return action, probs, values

    def _calculate_loss(
        self,
        values: torch.Tensor,
        next_values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        probs: torch.Tensor,
        old_probs: torch.Tensor,
    ):
        advantages = self._calculate_advantage(values, next_values, rewards, dones)
        surrogate_loss = self._clipped_surrogate_objective(
            torch.log(probs), torch.log(old_probs), advantages
        )
        value_loss = self._value_objective(values, next_values, rewards, dones)
        entropy_loss = self._entropy_objective(probs)
        loss = torch.mean(
            surrogate_loss
            - self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )
        return -loss

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

    def _value_objective(
        self,
        values: torch.Tensor,
        next_values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        value_targets = torch.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            value_targets[t] = rewards[t] + self.gamma * next_values[t] * (1 - dones[t])
        return (values - value_targets) ** 2

    def _entropy_objective(self, probs: torch.Tensor):
        dist = torch.distributions.Categorical(probs)
        entropy = dist.entropy()
        return entropy

    def _calculate_advantage(
        self,
        values: torch.Tensor,
        next_values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ):
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            next_value = next_values[t] * (1 - dones[t])
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        return advantages

    def _calculate_value_targets(
        self, rewards: torch.Tensor, next_values: torch.Tensor, dones: torch.Tensor
    ):
        value_targets = torch.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            value_targets[t] = rewards[t] + self.gamma * next_values[t] * (1 - dones[t])
        return value_targets
