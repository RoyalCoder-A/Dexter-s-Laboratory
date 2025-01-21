import torch

from attention_is_all_you_need.implementation.src.layers.attention import (
    MultiheadAttention,
)


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


class EncoderSublayer(torch.nn.Module):
    def __init__(self, d_model: int, dff: int, heads: int, device: str):
        super().__init__()
        self.multi_head_attention = MultiheadAttention(d_model, heads, device)
        self.multi_head_normalizer = torch.nn.LayerNorm(d_model)
        self.multi_head_dropout = torch.nn.Dropout(0.1)
        self.feed_forward = FeedForward(d_model, dff, device)
        self.feed_forward_normalizer = torch.nn.LayerNorm(d_model)
        self.feed_forward_dropout = torch.nn.Dropout(0.1)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.multi_head_normalizer(
            self.multi_head_dropout(self.multi_head_attention(x)) + x
        )
        x = self.feed_forward_normalizer(
            self.feed_forward_dropout(self.feed_forward(x)) + x
        )
        return x


class Encoder(torch.nn.Module):

    def __init__(self, n: int, d_model: int, dff: int, heads: int, device: str):
        super().__init__()
        self.sub_layers = torch.nn.Sequential(
            *[EncoderSublayer(d_model, dff, heads, device) for _ in range(n)]
        )
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sub_layers(x)
