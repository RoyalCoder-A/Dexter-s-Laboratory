import torch


class PreLayer(torch.nn.Module):
    def __init__(
        self,
        dictionary_size: int,
        d_model: int,
        max_length: int,
        drouput_prob: float,
        device: str,
    ):
        super().__init__()
        self.d_model = d_model
        self.embeddings = torch.nn.Embedding(dictionary_size, d_model)
        self.pe = _create_positional_encoding(d_model, max_length).to(device)
        self.dropout = torch.nn.Dropout(drouput_prob)
        self.to(device)

    def forward(self, x):
        x = self.embeddings(x) * torch.sqrt(
            torch.tensor(self.d_model).type(torch.int32)
        )
        x = x + self.pe
        x = self.dropout(x)
        return x


def _create_positional_encoding(d_model: int, max_len: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float()
        * (-torch.log(torch.tensor(10000.0)) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe
