import torch


class RModel(torch.nn.Module):
    def __init__(
        self, num_state: int, num_actions: int, num_hidden: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_state, num_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_hidden, num_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_hidden, num_actions),
            torch.nn.Softmax(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
