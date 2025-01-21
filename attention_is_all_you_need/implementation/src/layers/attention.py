import torch


"""
1. Input data will be simple tensor: (32, 256, 512)
2. Will create qkv by passing to linear layer: (32, 256, 512, 3)
3. Split into heads: (32, 256, 8, 64, 3)
4. Apply attention: (32, 256, 8, 64, 1)
5. Concatenate heads: (32, 256, 512, 1)
6. Apply linear layer: (32, 256, 512)
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
