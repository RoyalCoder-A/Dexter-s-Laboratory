import torch

from attention_is_all_you_need.implementation.src.layers.attention import (
    DecoderAttention,
    MultiheadAttention,
)
from attention_is_all_you_need.implementation.src.layers.feed_forward import FeedForward


class DecoderSubLayer(torch.nn.Module):
    def __init__(self, d_model: int, dff: int, heads: int, device: str):
        super().__init__()
        self.masked_attention = MultiheadAttention(d_model, heads, device)
        self.masked_attention_normalizer = torch.nn.LayerNorm(d_model)
        self.masked_attention_dropout = torch.nn.Dropout(0.1)

        self.cross_attention = DecoderAttention(d_model, heads, device)
        self.cross_attention_normalizer = torch.nn.LayerNorm(d_model)
        self.cross_attention_dropout = torch.nn.Dropout(0.1)

        self.feed_forward = FeedForward(d_model, dff, device)
        self.feed_forward_normalizer = torch.nn.LayerNorm(d_model)
        self.feed_forward_dropout = torch.nn.Dropout(0.1)
        self.to(device)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, encoder_output: torch.Tensor
    ) -> torch.Tensor:
        x = self.masked_attention_normalizer(
            self.masked_attention_dropout(self.masked_attention(x, mask) + x)
        )
        x = self.cross_attention_normalizer(
            self.cross_attention_dropout(self.cross_attention(x, encoder_output) + x)
        )
        x = self.feed_forward_normalizer(
            self.feed_forward_dropout(self.feed_forward(x) + x)
        )
        return x
