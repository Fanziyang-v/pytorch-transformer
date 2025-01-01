from torch import nn, Tensor
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class Transformer(nn.Module):
    """Transformer model."""

    def __init__(
        self,
        enc_vocab_size: int,
        dec_vocab_size: int,
        max_len: int,
        d_model: int,
        d_hidden: int,
        n_heads: int,
        n_enc_layers: int,
        n_dec_layers: int,
        drop_prob: float,
    ) -> None:
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(
            enc_vocab_size=enc_vocab_size,
            max_len=max_len,
            d_model=d_model,
            d_hidden=d_hidden,
            n_heads=n_heads,
            n_layers=n_enc_layers,
            drop_prob=drop_prob,
        )
        self.decoder = TransformerDecoder(
            dec_vocab_size=dec_vocab_size,
            max_len=max_len,
            d_model=d_model,
            d_hidden=d_hidden,
            n_heads=n_heads,
            n_layers=n_dec_layers,
            drop_prob=drop_prob,
        )
        self.linear = nn.Linear(d_model, dec_vocab_size)

    def forward(
        self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor
    ) -> Tensor:
        memory = self.encoder(src, src_mask)
        out = self.decoder(tgt, memory, src_mask, tgt_mask)
        return self.linear(out)
