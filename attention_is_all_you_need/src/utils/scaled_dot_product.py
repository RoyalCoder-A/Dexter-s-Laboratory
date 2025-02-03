import math
import torch


class ScaledDotProduct(torch.nn.Module):

    def __init__(self, p_dropout: float = 0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(p_dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        args:
            q: (batch_size, num_heads, seq_len, d_k)
            k: (batch_size, num_heads, seq_len, d_k)
            v: (batch_size, num_heads, seq_len, d_v)
            mask: (batch_size, num_heads, seq_len, seq_len)
        returns:
            out: (batch_size, num_heads, seq_len, d_v)
        """
        d_k = q.size(-1)
        score = (
            q @ k.transpose(-2, -1) / math.sqrt(d_k)
        )  # (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            score = score.masked_fill(mask == 1, -1e9)
        score = self.dropout(torch.nn.functional.softmax(score, dim=-1))
        out = score @ v  # (batch_size, num_heads, seq_len, d_v)
        return out
