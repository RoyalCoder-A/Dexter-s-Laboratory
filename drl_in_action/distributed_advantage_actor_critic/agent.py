import numpy as np
import torch

from drl_in_action.distributed_advantage_actor_critic.model import ActorCriticModel


class TransitionMemory:
    def __init__(self, n: int, num_states: int):
        self.states = np.zeros((n, num_states), dtype=np.float64)
        self.actions = np.zeros((n,), dtype=np.int64)
        self.rewards = np.zeros((n,), dtype=np.float64)
        self.states_ = np.zeros((n, num_states), dtype=np.float64)
        self.terminal = np.zeros((n,), dtype=np.int64)
        self.counter = 0

    def remember(
        self,
        state: np.ndarray,
        action: np.ndarray,
        rewards: np.ndarray,
        state_: np.ndarray,
        done: bool,
    ):
        self.states[self.counter] = state
        self.actions[self.counter] = action
        self.rewards[self.counter] = rewards
        self.states_[self.counter] = state_
        self.terminal[self.counter] = int(done)
        self.counter += 1

    def reset(self):
        states = self.states[: self.counter]
        actions = self.actions[: self.counter]
        rewards = self.rewards[: self.counter]
        states_ = self.states_[: self.counter]
        terminals = self.terminal[: self.counter]
        self.counter = 0
        return states, actions, rewards, states_, terminals


class Agent:
    def __init__(
        self,
        model: ActorCriticModel,
        n: int,
        num_states: int,
        num_actions: int,
        discount: float,
        clc: float,
        device: str,
    ):
        self.model = model
        self.n = n
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount = discount
        self.device = device
        self.clc = clc
        self.memory = TransitionMemory(n, num_states)
        self.optim = torch.optim.Adam(self.model.parameters())

    def get_action(self, state: np.ndarray):
        self.model.eval()
        with torch.inference_mode():
            policy, _ = self.model(
                torch.from_numpy(state).view(1, self.num_states).to(self.device)
            )
            policy: torch.Tensor = policy.detach().cpu().view(self.num_actions)
        dist = torch.distributions.Categorical(logits=policy)
        action = dist.sample()
        return action.numpy()

    def step(self):
        if self.memory.counter < self.n:
            return
        self.model.train()
        self.optim.zero_grad()
        states, actions, rewards, _, terminals = self.memory.reset()
        states_tensor = torch.from_numpy(states).float().to(self.device)
        actions_tensor = torch.from_numpy(actions).long().to(self.device)
        rewards_tensor = torch.from_numpy(rewards).float().to(self.device)
        terminals_tensor = torch.from_numpy(terminals).long().to(self.device)
        indices = torch.arange(len(actions_tensor)).long().to(self.device)
        log_probs, state_values = self.model(states_tensor)
        states_values = state_values.view((state_values.shape[0],))
        returns = self._calculate_returns(
            rewards_tensor, state_values, terminals_tensor
        )
        policy_loss = (
            -1 * log_probs[indices, actions_tensor] * (returns - state_values.detach())
        )
        critic_loss = torch.pow(returns - state_values, 2)
        loss: torch.Tensor = policy_loss.sum() + self.clc * critic_loss.sum()
        loss.backward()
        self.optim.step()

    def _calculate_returns(
        self, rewards: torch.Tensor, values: torch.Tensor, terminals: torch.Tensor
    ) -> torch.Tensor:
        rewards_rev = rewards.flip(dims=(0,))
        values_rev = values.flip(dims=(0,))
        terminals_rev = terminals.flip(dims=(0,))
        returns_rev = torch.zeros_like(rewards_rev)
        return_ = values_rev[0]
        for i in range(len(rewards_rev)):
            return_ = rewards_rev[i] + self.discount * return_ * (terminals_rev[i] - 1)
            returns_rev[i] = return_
        return torch.nn.functional.normalize(returns_rev.flip(dims=(0,)), dim=0)
