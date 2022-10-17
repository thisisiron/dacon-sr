import torch
import torch.nn as nn


class Activation(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(x)
