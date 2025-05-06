from pathlib import Path
import numpy as np
import torch
from reinforcement_learning.naive_q_learning.network import QNetwork
from torch import nn


class Agent:
    def __init__(
        self,
        q_network: QNetwork,
        device: str,
        eps: float,
        eps_min: float,
        eps_decay: float,
        n_actions: int,
        states_dim: tuple[int, ...],
        discount: float,
        checkpoint_path: Path,
    ) -> None:
        self.q_network = q_network
        self.loss_fn = nn.MSELoss().to(device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.n_actions = n_actions
        self.states_dim = states_dim
        self.discount = discount
        self.device = device
        self.checkpoint_path = checkpoint_path

    def get_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        self.q_network.eval()
        with torch.inference_mode():
            states_tensor = torch.from_numpy(state).float()
            q_values = (
                self.q_network(states_tensor.view(1, *self.states_dim).to(self.device))
                .cpu()
                .numpy()
            )
        return int(np.argmax(q_values[0]))

    def update_q_value(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.q_network.train()
        action_values = self.q_network(
            torch.from_numpy(state).float().view(1, *self.states_dim).to(self.device)
        )[0]
        next_action_values = self.q_network(
            torch.from_numpy(next_state)
            .float()
            .view(1, *self.states_dim)
            .to(self.device)
        )[0]
        if done:
            target = reward
        else:
            target = reward + self.discount * torch.max(next_action_values)
        loss = self.loss_fn(
            action_values[action], torch.tensor(target).float().to(self.device)
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.eps = max(self.eps - self.eps_decay, self.eps_min)

    def save(self):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_network.state_dict(), self.checkpoint_path)

    def load(self):
        self.q_network.load_state_dict(torch.load(self.checkpoint_path))
