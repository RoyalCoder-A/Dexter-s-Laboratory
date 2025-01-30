import torch


class MultiheadAttention(torch.nn.Module):
    def __init__(self, d_model: int, heads: int, device: str):
        super().__init__()
        assert (
            d_model % heads == 0
        ), f"d_model ({d_model}) must be divisible by heads ({heads})"
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        self.qkv = torch.nn.Linear(d_model, d_model * 3)
        self.out = torch.nn.Linear(d_model, d_model)
        self.to(device)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: [batch_size, seq_length, d_model]
        """
        if mask is not None and mask.dim == 3:  # [batch_size, 1, seq_length]
            mask = mask.unsqueeze(1)  # [batch_size, 1, 1, seq_length]
        batch_size = x.size(0)
        seq_length = x.size(1)
        qkv = self.qkv(x).view(batch_size, self.heads, seq_length, self.d_k, 3)
        q, k, v = qkv[:, :, :, :, 0], qkv[:, :, :, :, 1], qkv[:, :, :, :, 2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_k, dtype=torch.float32)
        )  # [batch_size, heads, seq_length, seq_length]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)  # [batch_size, heads, seq_length, d_k]
        context = context.transpose(
            1, 2
        ).contiguous()  # [batch_size, seq_length, heads, d_k]
        context = context.view(batch_size, seq_length, self.d_model)
        return self.out(context)


class CrossAttention(torch.nn.Module):
    def __init__(self, d_model: int, heads: int, device: str):
        super().__init__()
        assert (
            d_model % heads == 0
        ), f"d_model ({d_model}) must be divisible by heads ({heads})"
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        self.kv = torch.nn.Linear(d_model, d_model * 2)
        self.out = torch.nn.Linear(d_model, d_model)
        self.to(device)

    def forward(
        self,
        q: torch.Tensor,
        encoder_output: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        q: [batch_size, decoder_seq_length, d_model]
        encoder_output: [batch_size, encoder_seq_length, d_model]
        """
        if mask is not None and mask.dim == 3:  # [batch_size, 1, decoder_seq_length]
            mask = mask.unsqueeze(1)  # [batch_size, 1, 1, decoder_seq_length]
        batch_size = encoder_output.size(0)
        encoder_seq_length = encoder_output.size(1)
        decoder_seq_length = q.size(1)
        kv = self.kv(encoder_output).view(
            batch_size, self.heads, encoder_seq_length, self.d_k, 2
        )  # [batch_size, heads, encoder_seq_length, d_k, 2]
        k, v = (
            kv[:, :, :, :, 0],
            kv[:, :, :, :, 1],
        )  # [batch_size, heads, encoder_seq_length, d_k]
        q = q.view(
            batch_size, decoder_seq_length, self.heads, self.d_k
        )  # [batch_size, decoder_seq_length, heads, d_k]
        q = q.transpose(1, 2)  # [batch_size, heads, decoder_seq_length, d_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_k, dtype=torch.float32)
        )  # [batch_size, heads, decoder_seq_length, encoder_seq_length]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.nn.functional.softmax(
            scores, dim=-1
        )  # [batch_size, heads, decoder_seq_length, encoder_seq_length]
        context = torch.matmul(attn, v)  # [batch_size, heads, decoder_seq_length, d_k]
        context = context.transpose(
            1, 2
        ).contiguous()  # [batch_size, decoder_seq_length, heads, d_k]
        context = context.view(
            batch_size, decoder_seq_length, self.d_model
        )  # [batch_size, decoder_seq_length, d_model]
        return self.out(context)  # [batch_size, decoder_seq_length, d_model]
