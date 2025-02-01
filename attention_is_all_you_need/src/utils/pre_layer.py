import math
import torch

from attention_is_all_you_need.src.utils.positional_encoding import PositionalEncoding


class PreLayer(torch.nn.Module):
    def __init__(self, dict_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(dict_size, d_model)
        self.embedding.weight.data.normal_(0, 0.1)
        self.pe = PositionalEncoding(d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) * self.scale
        x = self.pe(x)
        return x
