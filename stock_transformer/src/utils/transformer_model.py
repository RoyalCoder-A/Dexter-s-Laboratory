import torch

from attention_is_all_you_need.src.utils.decoder import Decoder
from attention_is_all_you_need.src.utils.encoder import Encoder
from stock_transformer.src.utils.pre_layer import PreLayer


class TransformerModel(torch.nn.Module):

    def __init__(
        self,
        feature_size: int,
        d_model: int,
        n: int,
        dff: int,
        n_heads: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.pre_layer = PreLayer(feature_size, d_model)
        self.encoder = Encoder(n, d_model, dff, n_heads, p_dropout)
        self.decoder = Decoder(n, d_model, dff, n_heads, p_dropout)
        self.head = torch.nn.Linear(d_model, feature_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize the transformer parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(
        self, encoder_input: torch.Tensor, decoder_input: torch.Tensor
    ) -> torch.Tensor:
        src_mask, tgt_mask = self.generate_mask(encoder_input, decoder_input)
        encoder_input = self.pre_layer(encoder_input)
        encoder_output = self.encoder(encoder_input, src_mask)
        decoder_input = self.pre_layer(decoder_input)
        decoder_output = self.decoder(decoder_input, encoder_output, src_mask, tgt_mask)
        return self.head(decoder_output)

    def generate_mask(self, src: torch.Tensor, tgt: torch.Tensor):
        device = src.device

        # For multi-feature data, check if ALL features are non-zero (or use any() based on your needs)
        # This creates a sequence-level mask of shape (batch, seq_len)
        src_seq_mask = (src != 0).all(
            dim=-1
        )  # (batch, seq_len) - all features must be valid
        tgt_seq_mask = (tgt != 0).all(
            dim=-1
        )  # (batch, seq_len) - all features must be valid

        # Alternative: use any() if you want to mask only when ALL features are zero
        # src_seq_mask = (src != 0).any(dim=-1)  # (batch, seq_len)
        # tgt_seq_mask = (tgt != 0).any(dim=-1)  # (batch, seq_len)

        # Reshape for attention mechanism
        src_mask = src_seq_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
        tgt_mask = tgt_seq_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)

        seq_length = tgt.size(1)
        nopeak_mask = (
            (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1))
            .bool()
            .to(device)
        )  # (1, seq_length, seq_length)

        # Combine target mask with causal mask - now dimensions are compatible
        tgt_mask = tgt_mask & nopeak_mask  # (batch, 1, seq_len, seq_len)

        return src_mask, tgt_mask
