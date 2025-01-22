import torch

from attention_is_all_you_need.implementation.src.layers.decoder import Decoder
from attention_is_all_you_need.implementation.src.layers.encoder import Encoder
from attention_is_all_you_need.implementation.src.layers.pre_layer import PreLayer


class TransformerModel(torch.nn.Module):
    def __init__(
        self,
        dict_size: int,
        max_length: int,
        n: int,
        d_model: int,
        dff: int,
        heads: int,
        device: str,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.pre_layer = PreLayer(dict_size, d_model, max_length, 0.1, device)
        self.encoder = Encoder(n, d_model, dff, heads, device)
        self.decoder = Decoder(n, d_model, dff, heads, device)
        self.out = torch.nn.Linear(d_model, dict_size)
        self.device = device
        self.to(device)

    def create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """Create padding mask for the input sequence."""
        return (seq != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def create_look_ahead_mask(self, size: int) -> torch.Tensor:
        """Create look-ahead mask for the decoder."""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0

    def forward(
        self, encoder_input: torch.Tensor, decoder_input: torch.Tensor
    ) -> torch.Tensor:
        src_mask = self.create_padding_mask(encoder_input).to(self.device)
        tgt_mask = self.create_look_ahead_mask(decoder_input.size(1)).to(self.device)
        tgt_padding_mask = self.create_padding_mask(decoder_input).to(self.device)
        encoder_input = self.pre_layer(encoder_input)
        decoder_input = self.pre_layer(decoder_input)
        encoder_output = self.encoder(encoder_input, src_mask)
        decoder_output = self.decoder(
            decoder_input, src_mask, tgt_padding_mask, tgt_mask, encoder_output
        )
        return self.out(decoder_output)
