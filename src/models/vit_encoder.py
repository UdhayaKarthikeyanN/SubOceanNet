"""ViT encoder - EXPERIMENTAL STUB.

Implements a minimal working patch-embed + Transformer stack so the config
switch `model.encoder_type: "vit"` runs end-to-end, but it is NOT tuned.
TODO(items marked for future work):
  * sinusoidal/RoPE positional encoding instead of learned absolute
  * windowed / shifted attention to keep cost linear in region size
  * CLS-token pooling + intermediate layer supervision
  * pretraining on masked patch reconstruction before profile regression
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseEncoder


class ViTEncoder(BaseEncoder):
    def __init__(self, latent_dim: int = 256, embed_dim: int = 128,
                 patch_size: int = 3, num_layers: int = 3, num_heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.2,
                 in_channels: int = 7):
        super().__init__(in_channels=in_channels, latent_dim=latent_dim)
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        # TODO: replace with fixed sin-cos or rotary embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim))  # resized lazily
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio), dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)      # (B, N, D)
        n_tok = tokens.shape[1]
        if self.pos_embed.shape[1] < n_tok:                          # lazy grow
            device = self.pos_embed.device
            extra = torch.randn(1, n_tok - self.pos_embed.shape[1],
                                self.pos_embed.shape[2], device=device) * 0.02
            self.pos_embed = nn.Parameter(torch.cat([self.pos_embed, extra], dim=1))
        tokens = tokens + self.pos_embed[:, :n_tok]
        tokens = self.norm(self.blocks(tokens))
        pooled = tokens.mean(dim=1)
        return self.head(pooled)
