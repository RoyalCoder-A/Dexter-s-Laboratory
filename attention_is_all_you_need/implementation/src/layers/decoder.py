import torch

from attention_is_all_you_need.implementation.src.layers.attention import (
    CrossAttention,
    MultiheadAttention,
)
from attention_is_all_you_need.implementation.src.layers.feed_forward import FeedForward


class DecoderSubLayer(torch.nn.Module):
    def __init__(self, d_model: int, dff: int, heads: int, device: str):
        super().__init__()
        self.masked_attention = MultiheadAttention(d_model, heads, device)
        self.masked_attention_normalizer = torch.nn.LayerNorm(d_model)
        self.masked_attention_dropout = torch.nn.Dropout(0.1)

        self.cross_attention = CrossAttention(d_model, heads, device)
        self.cross_attention_normalizer = torch.nn.LayerNorm(d_model)
        self.cross_attention_dropout = torch.nn.Dropout(0.1)

        self.feed_forward = FeedForward(d_model, dff, device)
        self.feed_forward_normalizer = torch.nn.LayerNorm(d_model)
        self.feed_forward_dropout = torch.nn.Dropout(0.1)
        self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        decoder_padding_mask: torch.Tensor,
        lookahead_mask: torch.Tensor,
        encoder_output: torch.Tensor,
    ) -> torch.Tensor:
        x = self.masked_attention_normalizer(
            self.masked_attention_dropout(
                self.masked_attention(x, lookahead_mask & decoder_padding_mask) + x
            )
        )
        x = self.cross_attention_normalizer(
            self.cross_attention_dropout(
                self.cross_attention(x, encoder_output, encoder_padding_mask) + x
            )
        )
        x = self.feed_forward_normalizer(
            self.feed_forward_dropout(self.feed_forward(x) + x)
        )
        return x


class Decoder(torch.nn.Module):

    def __init__(self, n: int, d_model: int, dff: int, heads: int, device: str):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [DecoderSubLayer(d_model, dff, heads, device) for _ in range(n)]
        )
        self.final_normalizer = torch.nn.LayerNorm(d_model)
        self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        decoder_padding_mask: torch.Tensor,
        look_ahead_mask: torch.Tensor,
        encoder_output: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                encoder_padding_mask,
                decoder_padding_mask,
                look_ahead_mask,
                encoder_output,
            )
        x = self.final_normalizer(x)
        return x
