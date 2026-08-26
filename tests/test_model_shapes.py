"""Model shape / contract tests: encoder -> embedding -> temperature-only head."""
from __future__ import annotations

import numpy as np
import torch


def test_cnn_encoder_latent_shape(test_env):
    from src.models import build_model

    model = build_model(test_env)
    x = torch.randn(4, 7, 9, 9)
    with torch.no_grad():
        z = model.encoder(x)
    assert z.shape == (4, test_env["model"]["latent_dim"])


def test_full_forward_outputs_only_temperature(test_env):
    from src.models import build_model

    model = build_model(test_env)
    model.eval()
    x = torch.randn(3, 7, 9, 9)
    with torch.no_grad():
        y = model(x)
    nz = len(test_env["depths_m"])
    assert y.shape == (3, nz)
    # single output head -> nothing but temperature can leave the network
    assert model.decoder.out_dim == nz
    assert not any(k.startswith(("sst", "sss", "sla", "cur", "wind"))
                   for k in dict(model.named_children()))


def test_vit_encoder_forward(test_env):
    from src.models import ViTEncoder

    enc = ViTEncoder(latent_dim=32, embed_dim=24, patch_size=3,
                     num_layers=1, num_heads=2, mlp_ratio=2.0, dropout=0.1)
    enc.eval()
    with torch.no_grad():
        z = enc(torch.randn(2, 7, 9, 9))
    assert z.shape == (2, 32)


def test_mc_dropout_gives_stochastic_uncertainty(test_env):
    from src.models import build_model, enable_mc_dropout

    model = build_model(test_env)   # dropout p=0.25 in tiny config
    model.eval()
    enable_mc_dropout(model)
    x = torch.randn(8, 7, 9, 9)
    outs = [model(x).detach().numpy() for _ in range(6)]
    spread = np.std(np.stack(outs), axis=0).mean()
    assert spread > 1e-6, "MC dropout produced identical passes; uncertainty would be zero"


def test_patch_extraction_matches_manual_window(test_env):
    """sliding-window patches must equal manually padded slices."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.inference import Predictor

    pred = Predictor.get(test_env)
    tensor = np.random.default_rng(0).normal(size=(7, 12, 12)).astype(np.float32)
    patches = pred._patches_full_grid(tensor)
    k = pred.patch
    r = k // 2
    padded = np.pad(tensor, ((0, 0), (r, r), (r, r)), mode="edge")
    manual = padded[:, 0:0 + k, 0:0 + k]
    assert patches.shape == (144, 7, k, k)
    assert np.allclose(patches[0], manual)

    # gradient direction: patch (i,j) centred on cell (i,j)
    c = patches.reshape(12, 12, 7, k, k)[6, 6]
    assert np.allclose(c[:, r, r], tensor[:, 6, 6])
