import numpy as np
import torch
from drl_in_action.reinforce.model import RModel


class ReinforceAgent:
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        num_hidden: int,
        discount: float,
        device: str,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_hidden = num_hidden
        self.discount = discount
        self.device = device

        self.actions = np.arange(num_actions)
        self.transitions: list[
            tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]
        ] = []
        self.pi = RModel(num_states, num_actions, num_hidden).to(device)
        self.opt = torch.optim.Adam(self.pi.parameters())
        self.step_count = 0

    def get_action(self, state: np.ndarray) -> np.ndarray:
        self.pi.eval()
        with torch.inference_mode():
            preds: torch.Tensor = self.pi(
                torch.from_numpy(state).to(self.device).float().view(1, self.num_states)
            )  # (1, num_actions)
        prob = preds.cpu().view(self.num_actions).numpy()
        action = np.random.choice(self.actions, p=prob)
        return action

    def remember(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        state_: np.ndarray,
        terminated: bool,
    ):
        self.transitions.append((state, action, reward, state_, terminated))

    def reset(self):
        if not self.transitions:
            return None
        self.opt.zero_grad()
        states, actions, rewards, _, _ = zip(*self.transitions)
        returns = self._get_returns(rewards)
        self.pi.train()
        preds: torch.Tensor = self.pi(
            torch.tensor(np.array(states)).float().to(self.device)
        )  # (transition_size, action_size)
        actions_tensor = torch.from_numpy(np.array(actions)).long().to(self.device)
        indices = torch.arange(preds.shape[0]).long().to(self.device)
        action_probs = preds[indices, actions_tensor]  # (transition_size,)
        target = -1 * torch.sum(returns * torch.log(action_probs)).to(
            self.device
        )  # (1,)
        target.backward()
        self.opt.step()
        self.transitions = []
        return target.detach().cpu().numpy()

    def step(self):
        self.step_count += 1

    def _get_returns(self, rewards: list[float]) -> torch.Tensor:
        if not rewards:
            return []
        delta = rewards[-1]
        returns: list[float] = [delta]
        for r in rewards[::-1][1:]:
            delta = r + self.discount * delta
            returns.append(delta)
        returns_arr = np.array(returns[::-1])
        return (
            torch.from_numpy(
                (returns_arr - returns_arr.mean()) / (returns_arr.std() + 1e-12)
            )
            .float()
            .to(self.device)
        )
