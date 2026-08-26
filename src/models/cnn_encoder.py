"""Default CNN encoder: Conv blocks 7 -> 32 -> 64 -> 128 (3x3, BN, ReLU)
with stride-2 downsampling, global average+max pooling, FC -> latent (256).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseEncoder


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class CNNEncoder(BaseEncoder):
    def __init__(self, latent_dim: int = 256,
                 channels: list[int] | None = None, dropout: float = 0.2):
        super().__init__(in_channels=7, latent_dim=latent_dim)
        channels = channels or [32, 64, 128]
        blocks, cin = [], self.in_channels
        for cout in channels:
            blocks.append(_conv_block(cin, cout))
            cin = cout
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(2 * cin, latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        avg = h.mean(dim=(2, 3))
        mx = h.amax(dim=(2, 3))
        return self.head(torch.cat([avg, mx], dim=1))
