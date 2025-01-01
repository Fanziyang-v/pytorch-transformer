import torch
from torch import nn, Tensor


class TransformerEmbedding(nn.Module):
    """Transformer Embedding."""

    def __init__(
        self, vocab_size: int, d_model: int, max_len: int, drop_prob: float
    ) -> None:
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionEmbedding(d_model, max_len)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        # input tensor of shape (batch_size, seq_len)
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.dropout(tok_emb + pos_emb)


class PositionEmbedding(nn.Module):
    """Positional Encoding."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super(PositionEmbedding, self).__init__()
        self.pos_encoding = torch.zeros(max_len, d_model, requires_grad=False)
        factor = 10000 ** (torch.arange(0, d_model, step=2) / d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        self.pos_encoding[:, 0::2] = torch.sin(pos / factor)
        self.pos_encoding[:, 1::2] = torch.cos(pos / factor)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size()[1]
        return self.pos_encoding[:seq_len, :].unsqueeze(0)


class TokenEmbedding(nn.Embedding):
    """Token Embedding."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=0)
