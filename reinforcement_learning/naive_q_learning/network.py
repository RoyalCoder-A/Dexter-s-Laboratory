from torch import nn
import torch


class QNetwork(nn.Module):
    def __init__(self, states_dim: tuple[int, ...], n_actions: int) -> None:
        super().__init__()
        input_dim = 1
        for dim in states_dim:
            input_dim *= dim
        self.input_dim = input_dim
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = state.view(-1, self.input_dim)
        return self.linear(state)
