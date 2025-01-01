from torch import nn, Tensor
from functools import partial

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        # input tensor of shape (batch_size, seq_len, d_model)
        # 1. linear transformation
        q, k, v = self.proj_q(q), self.proj_k(k), self.proj_v(v)
        # 2. split tensor by the number of heads
        q, k, v = map(partial(_split, n_heads=self.n_heads), (q, k, v))
        # 3. scaled dot-product attention
        out = self.attention(q, k, v, mask)
        # 4. concatenate heads
        out = _concat(out)
        # 5. linear transformation
        return self.proj_o(out)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        # input tensor of shape (batch_size, n_heads, seq_len, d_head)
        d_k = k.size()[3]
        k_t = k.transpose(2, 3)
        # 1. compute attention score
        score: Tensor = (q @ k_t) * d_k**-0.5
        # 2. apply mask(optional)
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))
        # 3. compute attention weights
        attn = self.softmax(score)
        # 4. compute attention output
        out = attn @ v
        return out


def _split(tensor: Tensor, n_heads: int) -> Tensor:
    """Split tensor by the number of heads."""
    batch_size, seq_len = tensor.size()[:2]
    d_model = tensor.size()[2]
    d_head = d_model // n_heads
    return tensor.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)


def _concat(tensor: Tensor) -> Tensor:
    """Concatenate tensor after splitting."""
    batch_size, n_heads, seq_len, d_head = tensor.size()
    d_model = n_heads * d_head
    return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
