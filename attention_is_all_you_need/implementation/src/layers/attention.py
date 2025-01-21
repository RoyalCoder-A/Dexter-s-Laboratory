import torch


"""
Typical multi-head attention shapes:

Let:
    B = batch_size
    S = sequence_length
    H = number_of_heads
    d_model = model dimension
    d_k = d_model / H

1. Input to attention (x): [B, S, d_model]
2. Q, K, V after linear layers: [B, S, d_model]
3. Split heads (reshape): [B, S, H, d_k]
4. Transpose for attention: [B, H, S, d_k]
5. Compute scores = Q x K^T / sqrt(d_k): [B, H, S, S]
6. Apply mask and softmax over last dimension: [B, H, S, S]
7. Multiply by V: [B, H, S, d_k]
8. Transpose/reshape to merge heads: [B, S, d_model]
"""


"""
Padding Mask in Multi-Head Attention

Purpose:
    - Ensures that attention is not paid to padding tokens.
    - Typically, a binary mask of shape [B, S] (1 = valid token, 0 = padding).

Usage:
    1. Original mask shape: [B, S]
       - B = batch_size
       - S = sequence_length
    2. Broadcast it to match attention score shapes:
       - For self-attention: [B, 1, 1, S] -> eventually [B, heads, S, S] via broadcast.
       - For cross-attention: similar expansion, e.g. [B, 1, S_src, S_tgt].
    3. Apply to attention logits (scores) of shape [B, heads, S, S] by:
       scores = scores.masked_fill(mask == 0, float('-inf'))
    4. Softmax over the last dimension so that all positions marked as padding get 0 weight.

Result:
    - The model ignores padded positions when calculating attention.
"""


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
        qkv = self.qkv(x).view(x.size(0), x.size(1), self.heads, self.d_k, 3)
        q, k, v = qkv[:, :, :, :, 0], qkv[:, :, :, :, 1], qkv[:, :, :, :, 2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_k, dtype=torch.float32)
        )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        return self.out(context.view(x.size(0), x.size(1), self.d_model))


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
        kv = self.kv(encoder_output).view(
            encoder_output.size(0), encoder_output.size(1), self.heads, self.d_k, 2
        )
        k, v = kv[:, :, :, :, 0], kv[:, :, :, :, 1]
        q = q.view(q.size(0), q.size(1), self.heads, self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_k, dtype=torch.float32)
        )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        return self.out(context.view(q.size(0), q.size(1), self.d_model))
