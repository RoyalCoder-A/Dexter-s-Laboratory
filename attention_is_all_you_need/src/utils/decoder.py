import torch

from attention_is_all_you_need.src.utils.feed_forward import FeedForward
from attention_is_all_you_need.src.utils.multi_head_attention import MultiheadAttention


class DecoderSelfAttentionSubLayer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        p_dropout: float = 0.1,
    ):
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


class DecoderCrossAttentionSubLayer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        p_dropout: float = 0.1,
    ):
        super().__init__()
        self.sub_layer = MultiheadAttention(d_model, n_heads, p_dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(p_dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        args:
            x: (batch_size, seq_len, d_model)
        return:
            out: (batch_size, seq_len, d_model
        """
        residual = x
        x = self.layer_norm(x)
        x = self.sub_layer(encoder_output, x, mask)
        x = self.dropout(x)
        return x + residual


class DecoderFeedForwardSubLayer(torch.nn.Module):
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


class DecoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        dff: int,
        n_heads: int,
        p_dropout: float = 0.1,
    ):
        super().__init__()
        self.layer1 = DecoderSelfAttentionSubLayer(d_model, n_heads, p_dropout)
        self.layer2 = DecoderCrossAttentionSubLayer(d_model, n_heads, p_dropout)
        self.layer3 = DecoderFeedForwardSubLayer(d_model, dff, p_dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_msk: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.layer1(x, tgt_mask)
        x = self.layer2(x, encoder_output, src_msk)
        x = self.layer3(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(
        self,
        n: int,
        d_model: int,
        dff: int,
        n_heads: int,
        p_dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [DecoderLayer(d_model, dff, n_heads, p_dropout) for _ in range(n)]
        )
        self.layer_norm = torch.nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.layer_norm(x)
