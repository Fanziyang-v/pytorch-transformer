from torch import nn, Tensor


class MLP(nn.Module):
    """Multi-Layer Perceptron with one hidden layer."""

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1) -> None:
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out
