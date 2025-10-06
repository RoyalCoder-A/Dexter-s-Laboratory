import numpy as np
import torch
from drl_in_action.dqn.model import QModel
from drl_in_action.utils.reply_buffer import ReplyBuffer


class DQNAgent:
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        num_hidden: int,
        memory_size: int,
        batch_size: int,
        target_replace: int,
        eps: float,
        eps_decay: float,
        eps_min: float,
        discount: float,
        device: str,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_hidden = num_hidden
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.discount = discount
        self.target_replace = target_replace
        self.device = device
        self.batch_size = batch_size
        self.step_count = 0
        self.memory = ReplyBuffer(self.num_states, memory_size)
        self.loss_fn = torch.nn.MSELoss().to(device)
        self.q = QModel(num_states, num_actions, num_hidden).to(device)
        self.opt = torch.optim.Adam(self.q.parameters())
        self.q_target = QModel(num_states, num_actions, num_hidden).to(device)
        self.q_target.load_state_dict(self.q.state_dict())

    def get_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.eps:
            return np.random.randint(0, self.num_actions)
        self.q.eval()
        with torch.inference_mode():
            q: torch.Tensor = self.q(
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            )
        return int(q.cpu().argmax(dim=1).squeeze().numpy())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        state_: np.ndarray,
        terminate: bool,
    ):
        self.memory.remember(state, action, reward, state_, terminate)

    def step(self):
        loss = None
        if self.memory.counter >= self.batch_size:
            loss = self._learn()
        self.step_count += 1
        if self.step_count % self.target_replace == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        self.eps = max(self.eps - self.eps_decay, self.eps_min)
        return loss

    def _learn(self):
        self.opt.zero_grad()
        states, actions, rewards, states_, terminates = self.memory.sample(
            self.batch_size
        )
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        states__tensor = torch.tensor(states_, dtype=torch.float32, device=self.device)
        terminates_tensor = torch.tensor(
            terminates, dtype=torch.float32, device=self.device
        )
        self.q_target.eval()
        with torch.inference_mode():
            q_: torch.Tensor = self.q_target(states__tensor)
        target = rewards_tensor + (
            self.discount * q_.max(dim=1).values * (1 - terminates_tensor)
        )
        self.q.train()
        q: torch.Tensor = self.q(states_tensor)
        q_pred = q[torch.arange(self.batch_size), actions]
        loss: torch.Tensor = self.loss_fn(
            q_pred,
            target,
        )
        loss.backward()
        self.opt.step()
        return float(loss.detach().cpu().numpy())
