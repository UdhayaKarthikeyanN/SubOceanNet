"""Decoder head: compact latent embedding -> temperature at the 15 depth levels.

This is the ONLY path from embedding to physical outputs. It produces exactly
``out_dim = len(depths_m)`` values per grid cell - never SST/SSS/SLA/currents/wind.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden: list[int] | None = None,
                 out_dim: int = 15, dropout: float = 0.2):
        super().__init__()
        hidden = hidden or [512, 256]
        layers: list[nn.Module] = [nn.LayerNorm(latent_dim)]
        din = latent_dim
        for h in hidden:
            layers += [nn.Linear(din, h), nn.GELU(), nn.Dropout(dropout)]
            din = h
        layers += [nn.Linear(din, out_dim)]   # temperature-only projection
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)                    # (B, out_dim) - degrees C, normalized space
