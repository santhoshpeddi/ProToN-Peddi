from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ViTEar(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        backbone_name: str = "vit_large_patch14_dinov2",
        pretrained_backbone: bool = True,
        backbone_checkpoint: str | Path | None = None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=bool(pretrained_backbone and backbone_checkpoint is None),
            num_classes=0,
        )

        if backbone_checkpoint is not None:
            self._load_backbone_checkpoint(backbone_checkpoint)

        self.fc = nn.Linear(self.backbone.num_features, embed_dim)
        self.embedding_dim = embed_dim

    def _load_backbone_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing backbone checkpoint: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict):
            if "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            elif "model" in state and isinstance(state["model"], dict):
                state = state["model"]

        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] Missing backbone keys when loading {checkpoint_path.name}: {missing[:5]}")
        if unexpected:
            print(f"[WARN] Unexpected backbone keys when loading {checkpoint_path.name}: {unexpected[:5]}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.fc(features)
        return F.normalize(embeddings, dim=1)
