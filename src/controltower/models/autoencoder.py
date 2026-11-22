"""Baseline one-hot autoencoder of Hemati et al. (2022).

Fully connected encoder/decoder (default 128-64-32-16-8-4-2 and mirror), LeakyReLU
(slope 0.4), tanh bottleneck, sigmoid output, Xavier-uniform init and BCE loss.
One-hot vectors are built per batch from integer codes, so the full one-hot
matrix (n x 5,825 for our Philadelphia encoding) never has to exist in memory.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _mlp(dims: list[int], last_act: nn.Module) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        lin = nn.Linear(dims[i], dims[i + 1])
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        layers.append(nn.LeakyReLU(0.4) if i < len(dims) - 2 else last_act)
    return nn.Sequential(*layers)


class OneHotAutoencoder(nn.Module):
    def __init__(self, cardinalities: list[int], n_num: int,
                 encoder_dim: list[int] = (128, 64, 32, 16, 8, 4, 2), bottleneck: str = "tanh"):
        super().__init__()
        self.cardinalities = list(cardinalities)
        self.n_num = n_num
        offsets = torch.tensor([0] + list(cardinalities[:-1])).cumsum(0)
        self.register_buffer("offsets", offsets, persistent=False)
        self.input_dim = int(sum(cardinalities)) + n_num
        enc = [self.input_dim] + list(encoder_dim)
        act = {"tanh": nn.Tanh(), "linear": nn.Identity(), "lrelu": nn.LeakyReLU(0.4)}[bottleneck]
        self.encoder = _mlp(enc, act)
        self.decoder = _mlp(enc[::-1], nn.Sigmoid())

    def to_input(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        x = torch.zeros(cat.shape[0], self.input_dim, device=cat.device)
        x.scatter_(1, cat + self.offsets, 1.0)
        x[:, self.input_dim - self.n_num:] = num
        return x

    def forward(self, cat, num):
        x = self.to_input(cat, num)
        z = self.encoder(x)
        return x, z, self.decoder(z)

    def loss(self, cat, num) -> torch.Tensor:
        """Mean BCE over all input dimensions (``torch.nn.BCELoss()`` in the reference code)."""
        x, _, x_hat = self(cat, num)
        return F.binary_cross_entropy(x_hat, x)

    @torch.no_grad()
    def field_scores(self, cat, num) -> torch.Tensor:
        """Per-record, per-field reconstruction error: (n, F + K) BCE summed inside each field."""
        x, _, x_hat = self(cat, num)
        bce = F.binary_cross_entropy(x_hat, x, reduction="none")
        sizes = self.cardinalities + [1] * self.n_num
        return torch.stack([s.sum(1) for s in torch.split(bce, sizes, dim=1)], dim=1)

    @torch.no_grad()
    def score(self, cat, num) -> torch.Tensor:
        """Record anomaly score = mean BCE over the input vector (the paper's L_Rec)."""
        return self.field_scores(cat, num).sum(1) / self.input_dim
