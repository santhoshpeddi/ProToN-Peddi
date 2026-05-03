from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ShapeAutoencoderConfig


class AutoencoderShapeIncorporated(nn.Module):
    """Autoencoder-style classifier over shape-derived distance features."""

    def __init__(self, num_classes: int, cfg: ShapeAutoencoderConfig):
        super().__init__()
        in_dim = cfg.num_feature_points
        enc1, enc2 = cfg.enc_dims
        dec1 = cfg.dec_dims[0] if len(cfg.dec_dims) > 0 else enc2

        self.enc_fc1 = nn.Linear(in_dim, enc1)
        self.enc_fc2 = nn.Linear(enc1, enc2)
        self.enc_ln = nn.LayerNorm(enc2)

        self.dec_fc1 = nn.Linear(enc2, dec1)
        self.dec_ln = nn.LayerNorm(dec1)
        self.out_fc = nn.Linear(dec1, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.enc_fc1(x))
        x = torch.tanh(self.enc_fc2(x))
        x = self.enc_ln(x)
        return x

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        h = torch.tanh(self.dec_fc1(z))
        h = self.dec_ln(h)
        return self.out_fc(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward_logits(x), dim=1)

    @torch.no_grad()
    def embed(self, x: torch.Tensor, l2_normalize: bool = True) -> torch.Tensor:
        z = self.encode(x)
        if l2_normalize:
            z = F.normalize(z, dim=1)
        return z
