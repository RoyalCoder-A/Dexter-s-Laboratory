import numpy as np
import torch
from drl_in_action.dist_dqn.memory import ReplyBuffer
from drl_in_action.dist_dqn.models import DistDqnModel
from drl_in_action.dist_dqn.probs import get_target_probs


class Agent:
    def __init__(
        self,
        num_state: int,
        num_action: int,
        num_hidden: int,
        num_support: int,
        limits: tuple[float, float],
        eps: float,
        eps_decay: float,
        eps_min: float,
        discount: float,
        memory_size: int,
        priority_size: int,
        network_replace: int,
        device: str,
    ):
        self.num_state = num_state
        self.num_action = num_action
        self.num_hidden = num_hidden
        self.num_support = num_support
        self.limits = limits
        self.device = device
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.discount = discount
        self.memory_size = memory_size
        self.priority_size = priority_size
        self.network_replace = network_replace
        self.counter = 0
        self.memory = ReplyBuffer(memory_size, num_state)
        self.q_model = DistDqnModel(
            num_state, num_hidden, num_action, num_support, limits[0], limits[1]
        ).to(device)
        self.q_target = DistDqnModel(
            num_state, num_hidden, num_action, num_support, limits[0], limits[1]
        ).to(device)
        self.q_target.load_state_dict(self.q_model.state_dict())
        self.opt = torch.optim.Adam(self.q_model.parameters())
        self.loss = torch.nn.CrossEntropyLoss().to(self.device)
        if device == "cuda":
            self.q_model.compile()
            self.q_target.compile()

    def get_actions(self, state: np.ndarray) -> int:
        if np.random.random() < self.eps:
            return np.random.randint(0, self.num_action)
        self.q_model.eval()
        with torch.inference_mode():
            q_values = self.q_model.get_q_values(
                torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            ).squeeze(0)
        return int(q_values.argmax(dim=-1).cpu().item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        state_: np.ndarray,
        done: bool,
    ) -> None:
        for _ in range(self.priority_size if done else 1):
            self.memory.remember(state, action, reward, state_, done)

    def step(self) -> None:
        self.counter += 1
        if self.memory.counter < self.memory_size:
            return
        states = torch.from_numpy(self.memory.states).float().to(self.device)
        rewards = torch.from_numpy(self.memory.rewards).float().to(self.device)
        states_ = torch.from_numpy(self.memory.states_).float().to(self.device)
        terminals = torch.from_numpy(self.memory.terminals).long().to(self.device)
        self.q_model.train()
        self.q_target.eval()
        self.opt.zero_grad()
        with torch.no_grad():
            next_probs = self.q_target.get_probs(states_)
            next_q_values = self.q_target.get_q_values(None, next_probs)
            next_greedy_actions = next_q_values.argmax(dim=-1)
            target_probs = get_target_probs(
                next_probs,
                next_greedy_actions,
                rewards,
                terminals,
                self.limits,
                self.discount,
            )
        probs: torch.Tensor = self.q_model(states)
        loss: torch.Tensor = self.loss(probs, target_probs)
        loss.backward()
        self.opt.step()
        self.eps = (
            self.eps - self.eps_decay if self.eps > self.eps_min else self.eps_min
        )
        if self.counter % self.network_replace == 0:
            self.q_target.load_state_dict(self.q_model.state_dict())
