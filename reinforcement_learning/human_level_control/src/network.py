import torch


class Network(torch.nn.Module):
    def __init__(self, input_shape: tuple[int, ...], n_actions: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 32, (8, 8), 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, (4, 4), 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, (3, 3), 1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(3136, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)
