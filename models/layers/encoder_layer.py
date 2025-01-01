import torch
from torch import nn, Tensor
from .mlp import MLP
from .attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    """Encoder layer."""

    def __init__(
        self, d_model: int, d_hidden: int, n_heads: int, drop_prob: float = 0.1
    ):
        super(EncoderLayer, self).__init__()
        # self-attention
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        # feed-forward network
        self.ffn = MLP(d_model, d_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # 1. self-attention
        shortcut = x
        out = self.attn(x, x, x, mask)
        out = self.norm1(shortcut + self.dropout1(out))

        # 2. feed-forward network
        shortcut = out
        out = self.ffn(out)
        out = self.norm2(shortcut + self.dropout2(out))
        return out
