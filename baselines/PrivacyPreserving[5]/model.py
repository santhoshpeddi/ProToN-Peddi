from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from config import TrainConfig


class DenseNet161Ear(nn.Module):
    """DenseNet-161 fine-tuned for multiclass ear classification."""

    def __init__(self, cfg: TrainConfig, num_classes: int):
        super().__init__()

        if cfg.use_local_imagenet_weights:
            weights_path = Path(cfg.imagenet_weights_path)
            if not weights_path.is_file():
                raise FileNotFoundError(
                    f"Local DenseNet-161 ImageNet weights not found: {weights_path}"
                )
            self.backbone = models.densenet161(weights=None)
            state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            elif isinstance(state, dict) and "model_state" in state and isinstance(state["model_state"], dict):
                state = state["model_state"]
            if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            self.backbone.load_state_dict(state, strict=True)
        else:
            try:
                weights = models.DenseNet161_Weights.IMAGENET1K_V1
                self.backbone = models.densenet161(weights=weights)
            except Exception:
                self.backbone = models.densenet161(weights=None)

        in_features = self.backbone.classifier.in_features
        self.embedding_dim = in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, l2_normalize: bool = False) -> torch.Tensor:
        features = self.backbone.features(x)
        out = F.relu(features)
        out = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        if l2_normalize:
            out = F.normalize(out, dim=1)
        return out
