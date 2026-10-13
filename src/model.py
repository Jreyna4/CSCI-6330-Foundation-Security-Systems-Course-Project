"""The single MLP used by every condition (held constant across experiments)."""

from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_sizes=(128, 64), dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(cfg, in_dim, num_classes):
    m = cfg["model"]
    return MLP(in_dim, num_classes, m["hidden_sizes"], m["dropout"])
