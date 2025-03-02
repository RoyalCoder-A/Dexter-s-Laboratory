import torch

from attention_is_all_you_need.src.utils.decoder import Decoder
from attention_is_all_you_need.src.utils.encoder import Encoder
from attention_is_all_you_need.src.utils.pre_layer import PreLayer


class TransformerModel(torch.nn.Module):

    def __init__(
        self,
        dict_size: int,
        d_model: int,
        n: int,
        dff: int,
        n_heads: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.pre_layer = PreLayer(dict_size, d_model)
        self.encoder = Encoder(n, d_model, dff, n_heads, p_dropout)
        self.decoder = Decoder(n, d_model, dff, n_heads, p_dropout)
        self.head = torch.nn.Linear(d_model, dict_size)
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
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)
        ).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
