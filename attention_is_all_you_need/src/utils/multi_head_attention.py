import torch

from attention_is_all_you_need.src.utils.scaled_dot_product import ScaledDotProduct


class MultiheadAttention(torch.nn.Module):

    def __init__(self, d_model: int, n_heads: int, device: str, p_dropout: float = 0.1):
        assert (
            d_model % n_heads == 0
        ), "Model dimension must be divisible by number of heads"
        super().__init__()
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.attention = ScaledDotProduct(device, p_dropout)
        self.n_heads = n_heads
        self.d_model = d_model
        self._reset_parameters()
        self.to(device)

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(
        self,
        kv_input: torch.Tensor,
        q_input: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        args:
            kv_input: (batch_size, seq_len, d_model)
            q_input: (batch_size, seq_len, d_model)
            mask: (1, 1, seq_len, seq_len)
        returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, q_seq_len, d_model = q_input.size()
        kv_seq_len = kv_input.size(1)

        k = self.k_proj(kv_input).view(
            batch_size, kv_seq_len, self.n_heads, -1
        )  # (batch_size, kv_seq_len, self.n_heads, dk)
        q = self.q_proj(q_input).view(
            batch_size, q_seq_len, self.n_heads, -1
        )  # (batch_size, q_seq_len, self.n_heads, dk)
        v = self.v_proj(kv_input).view(
            batch_size, kv_seq_len, self.n_heads, -1
        )  # (batch_size, kv_seq_len, self.n_heads, dv)

        k, q, v = map(
            lambda x: x.transpose(1, 2), (k, q, v)
        )  # (batch_size, self.n_heads, kv_seq_len, dk)
        attention = self.attention(
            q, k, v, mask
        )  # (batch_size, num_heads, seq_len, d_v)
        attention = (
            attention.transpose(1, 2).contiguous().view(batch_size, q_seq_len, -1)
        )  # (batch_size, q_seq_len, d_model)
        output = self.out_proj(attention)  # (batch_size, q_seq_len, d_model)
        return output
