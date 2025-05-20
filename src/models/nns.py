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
    def __init__(self, n_feature, n_output=1, n_hidden1=64, activation_hidden='relu', dropout=0.1):
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

class DBlock(nn.Module):
    def __init__(self, n_in, n_out, activation_fn_name="relu", dropout_rate=0.0, use_batch_norm=True):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        
        if activation_fn_name.lower() == "relu":
            self.activation_fn = nn.ReLU()
        elif activation_fn_name.lower() == "leaky_relu":
            self.activation_fn = nn.LeakyReLU()
        elif activation_fn_name.lower() == "elu":
            self.activation_fn = nn.ELU()
        elif activation_fn_name.lower() == "selu":
            self.activation_fn = nn.SELU()
        elif activation_fn_name.lower() == "gelu":
            self.activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn_name}")

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.bn = nn.BatchNorm1d(n_out) if use_batch_norm else None

    def forward(self, x):
        x = self.linear(x)
        # Apply BatchNorm before activation, common practice
        if self.bn:
            # Only apply BatchNorm if batch size > 1 during training.
            # In eval mode, BatchNorm uses running stats and works fine with batch size 1.
            if x.size(0) > 1 or not self.training:
                x = self.bn(x)
            # If batch_size is 1 and self.training is True, skip BN.
            # This prevents the "Expected more than 1 value per channel" error.
            # Note: Skipping BN for batch_size=1 means running_mean/var are not updated by these batches.
            # This is generally acceptable as single-sample estimates are very noisy.
        
        x = self.activation_fn(x)
        
        if self.dropout:
            x = self.dropout(x)
        return x

class DNet1(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_output, dropout=0.1, activation_fn="relu", **kw):
        super().__init__()
        self.blocks = nn.Sequential(
            DBlock(n_feature, n_hidden1, activation_fn, dropout),
            DBlock(n_hidden1, n_hidden2, activation_fn, dropout),
            DBlock(n_hidden2, n_hidden3, activation_fn, dropout),
            DBlock(n_hidden3, n_hidden4, activation_fn, dropout),
        )
        self.out = nn.Linear(n_hidden4, n_output)
    def forward(self, x):
        return self.out(self.blocks(x))

class DNet2(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5, n_output, dropout=0.1, activation_fn="relu", **kw):
        super().__init__()
        self.blocks = nn.Sequential(
            DBlock(n_feature, n_hidden1, activation_fn, dropout),
            DBlock(n_hidden1, n_hidden2, activation_fn, dropout),
            DBlock(n_hidden2, n_hidden3, activation_fn, dropout),
            DBlock(n_hidden3, n_hidden4, activation_fn, dropout),
            DBlock(n_hidden4, n_hidden5, activation_fn, dropout),
        )
        self.out = nn.Linear(n_hidden5, n_output)
    def forward(self, x):
        return self.out(self.blocks(x))

class DNet3(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5, n_output, dropout=0.1, activation_fn="relu", **kw):
        super().__init__()
        self.blocks = nn.Sequential(
            DBlock(n_feature, n_hidden1, activation_fn, dropout),
            DBlock(n_hidden1, n_hidden2, activation_fn, dropout),
            DBlock(n_hidden2, n_hidden3, activation_fn, dropout),
            DBlock(n_hidden3, n_hidden4, activation_fn, dropout),
            DBlock(n_hidden4, n_hidden5, activation_fn, dropout),
        )
        self.out = nn.Linear(n_hidden5, n_output)
    def forward(self, x):
        return self.out(self.blocks(x))