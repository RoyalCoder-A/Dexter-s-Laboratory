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
        causal_mask = self._generate_causal_mask(decoder_input).to(decoder_input.device)
        encoder_input = self.pre_layer(encoder_input)
        encoder_output = self.encoder(encoder_input)
        decoder_input = self.pre_layer(decoder_input)
        decoder_output = self.decoder(decoder_input, encoder_output, causal_mask)
        return self.head(decoder_output)

    def _generate_causal_mask(self, decoder_input: torch.Tensor) -> torch.Tensor:
        return (
            torch.triu(
                torch.ones(decoder_input.shape[1], decoder_input.shape[1]),
                diagonal=1,
            )
            .type(torch.int)
            .unsqueeze(0)
            .unsqueeze(0)
        ).bool()
