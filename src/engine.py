"""Training, prediction and weight get/set helpers shared by all conditions."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_epochs(model, loader, optimizer, epochs, device):
    """Train in place. Returns the mean training loss over the last epoch."""
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
            n += len(yb)
    return total / max(n, 1)


@torch.no_grad()
def predict_proba(model, X, device, batch_size=4096):
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)


def get_weights(model):
    """Model parameters as a list of numpy arrays (what a client 'sends')."""
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def set_weights(model, weights):
    keys = model.state_dict().keys()
    model.load_state_dict({k: torch.from_numpy(w) for k, w in zip(keys, weights)})
