import torch


class FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, dff: int, p_dropout: float = 0.1):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(d_model, dff),
            torch.nn.ReLU(),
            torch.nn.Dropout(p_dropout),
            torch.nn.Linear(dff, d_model),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
