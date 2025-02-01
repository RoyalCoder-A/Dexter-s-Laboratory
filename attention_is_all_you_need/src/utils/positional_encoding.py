import torch


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()

        # Create positional encoding matrix once during initialization
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        # Calculate sine and cosine positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (won't be updated during backprop)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [batch_size, seq_len, d_model]
        Returns:
            x: Tensor [batch_size, seq_len, d_model] with positional encoding added
        """
        return x + self.pe[:, : x.size(1)]
