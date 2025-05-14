import torch
from torch import nn

# helper – maps string → activation
_ACT = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "elu": nn.ELU(),
}

__all__ = [
    "Net1", "Net2", "Net3", "Net4", "Net5",
    "DNet1", "DNet2", "DNet3"
]

class _Base(nn.Module):
    def __init__(self, layers, dropout=0.0, act="relu"):
        super().__init__()
        act_fn = _ACT[act.lower()]
        seq = []
        for i in range(len(layers) - 2):
            seq.extend([nn.Linear(layers[i], layers[i+1]), act_fn, nn.Dropout(dropout)])
        seq.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)

class Net1(_Base):
    def __init__(self, n_feature, n_hidden1, n_output, dropout=0.0, activation_hidden="relu", **kw):
        super().__init__([n_feature, n_hidden1, n_output], dropout, activation_hidden)

class Net2(_Base):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output, dropout=0.0, activation_hidden="relu", **kw):
        super().__init__([n_feature, n_hidden1, n_hidden2, n_output], dropout, activation_hidden)

class Net3(_Base):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_output, dropout=0.0, activation_hidden="relu", **kw):
        super().__init__([n_feature, n_hidden1, n_hidden2, n_hidden3, n_output], dropout, activation_hidden)

class Net4(_Base):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_output, dropout=0.0, activation_hidden="relu", **kw):
        super().__init__([n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_output], dropout, activation_hidden)

class Net5(_Base):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5, n_output, dropout=0.0, activation_hidden="relu", **kw):
        super().__init__([n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5, n_output], dropout, activation_hidden)

class _DBlock(nn.Module):
    def __init__(self, inp, outp, dropout, act="relu"):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(inp, outp), nn.BatchNorm1d(outp), _ACT[act], nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.seq(x)

class DNet1(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_output, dropout=0.1, activation_fn="relu", **kw):
        super().__init__()
        self.blocks = nn.Sequential(
            _DBlock(n_feature, n_hidden1, dropout, activation_fn),
            _DBlock(n_hidden1, n_hidden2, dropout, activation_fn),
            _DBlock(n_hidden2, n_hidden3, dropout, activation_fn),
            _DBlock(n_hidden3, n_hidden4, dropout, activation_fn),
        )
        self.out = nn.Linear(n_hidden4, n_output)
    def forward(self, x):
        return self.out(self.blocks(x))

class DNet2(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_output, dropout=0.1, activation_fn="relu", **kw):
        super().__init__()
        self.blocks = nn.Sequential(
            _DBlock(n_feature, n_hidden1, dropout, activation_fn),
            _DBlock(n_hidden1, n_hidden2, dropout, activation_fn),
            _DBlock(n_hidden2, n_hidden3, dropout, activation_fn),
            _DBlock(n_hidden3, n_hidden4, dropout, activation_fn),
        )
        self.out = nn.Linear(n_hidden4, n_output)
    def forward(self, x):
        return self.out(self.blocks(x))

class DNet3(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5, n_output, dropout=0.1, activation_fn="relu", **kw):
        super().__init__()
        self.blocks = nn.Sequential(
            _DBlock(n_feature, n_hidden1, dropout, activation_fn),
            _DBlock(n_hidden1, n_hidden2, dropout, activation_fn),
            _DBlock(n_hidden2, n_hidden3, dropout, activation_fn),
            _DBlock(n_hidden3, n_hidden4, dropout, activation_fn),
            _DBlock(n_hidden4, n_hidden5, dropout, activation_fn),
        )
        self.out = nn.Linear(n_hidden5, n_output)
    def forward(self, x):
        return self.out(self.blocks(x))