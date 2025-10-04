import torch


class QModel(torch.nn.Module):
    def __init__(
        self, num_states: int, num_actions: int, num_hidden: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_hidden = num_hidden
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_states, num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
