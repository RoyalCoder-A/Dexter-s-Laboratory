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
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
