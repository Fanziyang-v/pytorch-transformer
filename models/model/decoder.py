from torch import nn, Tensor
from ..layers.embedding import TransformerEmbedding
from ..layers.decoder_layer import DecoderLayer


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        dec_vocab_size: int,
        max_len: int,
        n_layers: int,
        d_model: int,
        d_hidden: int,
        n_heads: int,
        drop_prob: float,
    ):
        super(TransformerDecoder, self).__init__()
        self.emb = TransformerEmbedding(dec_vocab_size, d_model, max_len, drop_prob)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, d_hidden, n_heads, drop_prob)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self, tgt: Tensor, src: Tensor, tgt_mask: Tensor, src_mask: Tensor
    ) -> Tensor:
        out = self.emb(tgt)
        for layer in self.layers:
            out = layer(out, src, tgt_mask, src_mask)
        return out
