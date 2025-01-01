from torch import nn, Tensor
from ..layers.embedding import TransformerEmbedding
from ..layers.encoder_layer import EncoderLayer


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        enc_vocab_size: int,
        max_len: int,
        n_layers: int,
        d_model: int,
        d_hidden: int,
        n_heads: int,
        drop_prob: float,
    ):
        super(TransformerEncoder, self).__init__()
        self.emb = TransformerEmbedding(enc_vocab_size, d_model, max_len, drop_prob)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, d_hidden, n_heads, drop_prob)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        out = self.emb(x)
        for layer in self.layers:
            out = layer(out, mask)
        return out
