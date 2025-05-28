from pathlib import Path
import numpy as np
import torch
from reinforcement_learning.human_level_control.src.memory import Memory
from reinforcement_learning.human_level_control.src.network import Network
from torch import nn


class Agent:
    def __init__(
        self,
        device: str,
        eps: float,
        eps_min: float,
        eps_decay: float,
        n_actions: int,
        states_dim: tuple[int, ...],
        discount: float,
        checkpoint_path: Path,
        replace: int,
        memory_size: int,
        batch_size: int,
        learning_rate: float
    ) -> None:
        self.q_main = Network(states_dim, n_actions).to(device)
        self.q_target = Network(states_dim, n_actions).to(device)
        if device == "cuda":
            self.q_main.compile()
            self.q_target.compile()
        self.q_target.load_state_dict(self.q_main.state_dict())
        self.memory = Memory(states_dim, max_size=memory_size)
        self.replace = replace
        self.loss_fn = nn.HuberLoss().to(device)
        self.optimizer = torch.optim.RMSprop(
            self.q_main.parameters(), lr=learning_rate)
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.n_actions = n_actions
        self.states_dim = states_dim
        self.discount = discount
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.step = 0
        self.batch_size = batch_size

    def replace_target(self):
        if self.step % self.replace != 0:
            return
        self.q_target.load_state_dict(self.q_main.state_dict())

    def get_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        self.q_main.eval()
        with torch.inference_mode():
            states_tensor = torch.from_numpy(state).float()
            q_values = (
                self.q_main(states_tensor.view(1, *self.states_dim).to(self.device))
                .cpu()
                .numpy()
            )
        return int(np.argmax(q_values[0]))

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminal: bool,
    ):
        self.memory.remember(
            state=state,
            action=action,
            reward=reward,
            state_=next_state,
            terminal=terminal,
        )

    def decrement_eps(self):
        self.eps = max(self.eps - self.eps_decay, self.eps_min)

    def learn(self):
        if self.step >= self.batch_size:
            self.optimizer.zero_grad()
            states, actions, rewards, states_, terminals = self.memory.sample(
                self.batch_size
            )
            states_t, actions_t, rewards_t, states__t, terminals_t = (
                torch.from_numpy(states).float().to(self.device),
                torch.from_numpy(actions).long().to(self.device),
                torch.from_numpy(rewards).float().to(self.device),
                torch.from_numpy(states_).float().to(self.device),
                torch.from_numpy(terminals).bool().to(self.device),
            )
            indices = torch.arange(self.batch_size, device=self.device)
            self.q_main.train()
            q_values = self.q_main(states_t)[indices, actions_t]
            self.q_target.eval()
            with torch.inference_mode():
                next_q_values = self.q_target(states__t).max(dim=1)[0]
                next_q_values[terminals_t] = 0.0
            target = rewards_t + self.discount * next_q_values
            loss = self.loss_fn(
                q_values, target
            )
            loss.backward()
            self.optimizer.step()
        self.replace_target()
        self.decrement_eps()
        self.step += 1

    def save(self):
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_main.state_dict(), self.checkpoint_path / "q_main.pth")
        torch.save(self.q_target.state_dict(), self.checkpoint_path / "q_target.pth")

    def load(self):
        self.q_main.load_state_dict(torch.load(self.checkpoint_path / "q_main.pth"))
        self.q_target.load_state_dict(torch.load(self.checkpoint_path / "q_target.pth"))
