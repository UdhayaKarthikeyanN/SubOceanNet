"""Abstract encoder contract + shared model utilities."""
from __future__ import annotations

import abc
from typing import Any

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, abc.ABC):
    """Contract for surface->latent encoders.

    Input : float tensor (B, 7, H, W) - the 7 satellite surface variables.
    Output: float tensor (B, latent_dim).
    """

    def __init__(self, in_channels: int = 7, latent_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @property
    def kind(self) -> str:
        return type(self).__name__.replace("Encoder", "").lower()


def enable_mc_dropout(model: nn.Module) -> None:
    """Keep dropout stochastic while the rest of the net stays in eval mode."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TemperatureModel(nn.Module):
    """Encoder -> latent embedding -> decoder producing ONLY temperature.

    The single output head emits exactly ``len(depths_m)`` channels; no input
    variable can ever appear in the output (see tests/test_model_shapes.py).
    """

    def __init__(self, encoder: BaseEncoder, decoder: nn.Module, depths: list[int]):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.depths = list(depths)
        self.input_variables: list[str] = []
        self.version: str = "0"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)  # (B, n_depths)

    def architecture_summary(self) -> str:
        enc = str(self.encoder).replace("\n", "\n  ")
        dec = str(self.decoder).replace("\n", "\n  ")
        return (
            f"TemperatureModel(output=temperature@{self.depths} m only)\n"
            f"ENCODER ({type(self.encoder).__name__}):\n  {enc}\n"
            f"DECODER:\n  {dec}"
        )


def build_model(cfg: dict[str, Any]) -> TemperatureModel:
    """Config-driven factory (model.encoder_type: 'cnn' | 'vit')."""
    from .cnn_encoder import CNNEncoder
    from .decoder import MLPDecoder
    from .vit_encoder import ViTEncoder

    mc = cfg["model"]
    etype = str(mc.get("encoder_type", "cnn")).lower()
    if etype == "cnn":
        encoder: BaseEncoder = CNNEncoder(
            latent_dim=int(mc["latent_dim"]),
            channels=list(mc.get("cnn_channels", [32, 64, 128])),
            dropout=float(mc.get("dropout", 0.2)),
        )
    elif etype == "vit":
        vit = mc.get("vit", {})
        encoder = ViTEncoder(
            latent_dim=int(mc["latent_dim"]),
            embed_dim=int(vit.get("embed_dim", 128)),
            patch_size=int(vit.get("patch_size", 3)),
            num_layers=int(vit.get("num_layers", 3)),
            num_heads=int(vit.get("num_heads", 4)),
            mlp_ratio=float(vit.get("mlp_ratio", 2.0)),
            dropout=float(mc.get("dropout", 0.2)),
        )
    else:
        raise ValueError(f"unknown encoder_type '{etype}' (expected 'cnn' or 'vit')")

    decoder = MLPDecoder(
        latent_dim=int(mc["latent_dim"]),
        hidden=list(mc.get("decoder_hidden", [512, 256])),
        out_dim=len(cfg["depths_m"]),
        dropout=float(mc.get("dropout", 0.2)),
    )
    return TemperatureModel(encoder, decoder, cfg["depths_m"])
