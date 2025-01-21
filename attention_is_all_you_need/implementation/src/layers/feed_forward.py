import torch


class FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: str):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(d_ff, d_model),
        )
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
