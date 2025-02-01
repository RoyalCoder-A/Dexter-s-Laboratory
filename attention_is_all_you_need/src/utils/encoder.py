import torch

from attention_is_all_you_need.src.utils.feed_forward import FeedForward
from attention_is_all_you_need.src.utils.multi_head_attention import MultiheadAttention


class EncoderAttentionSubLayer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        p_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sub_layer = MultiheadAttention(d_model, n_heads, p_dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(p_dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        args:
            x: (batch_size, seq_len, d_model)
        return:
            out: (batch_size, seq_len, d_model
        """
        residual = x
        x = self.layer_norm(x)
        x = self.sub_layer(x, x, mask)
        x = self.dropout(x)
        return x + residual


class EncoderFeedForwardSubLayer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        dff: int,
        p_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sub_layer = FeedForward(d_model, dff, p_dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: (batch_size, seq_len, d_model)
        return:
            out: (batch_size, seq_len, d_model
        """
        residual = x
        x = self.layer_norm(x)
        x = self.sub_layer(x)
        x = self.dropout(x)
        return x + residual


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, dff: int, n_heads: int, p_dropout: float = 0.1):
        super().__init__()
        self.layer1 = EncoderAttentionSubLayer(d_model, n_heads, p_dropout)
        self.layer2 = EncoderFeedForwardSubLayer(d_model, dff, p_dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.layer2(self.layer1(x, mask))


class Encoder(torch.nn.Module):
    def __init__(
        self, n: int, d_model: int, dff: int, n_heads: int, p_dropout: float = 0.1
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [EncoderLayer(d_model, dff, n_heads, p_dropout) for _ in range(n)]
        )
        self.layer_norm = torch.nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)
