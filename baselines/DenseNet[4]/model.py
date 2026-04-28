from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


@dataclass
class ModelConfig:
    num_classes: int
    use_local_pretrained: bool = False
    imagenet_weights_path: str | Path = 'weights/densenet161.pth'
    freeze_early_layers: bool = True
    unfreeze_last_denseblock: bool = True


def _load_local_imagenet_weights(backbone: nn.Module, path: str | Path) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Local ImageNet weights not found: {path}\nProvide DenseNet-161 ImageNet weights locally; automatic download is disabled."
        )
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if isinstance(state, dict) and any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    if missing:
        print(f'[WARN] Missing keys while loading {path.name}: {missing[:5]}')
    if unexpected:
        print(f'[WARN] Unexpected keys while loading {path.name}: {unexpected[:5]}')


class DenseNet161Ear(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        try:
            self.backbone = models.densenet161(weights=None)
        except TypeError:
            self.backbone = models.densenet161(pretrained=False)

        if cfg.use_local_pretrained:
            _load_local_imagenet_weights(self.backbone, cfg.imagenet_weights_path)

        in_features = self.backbone.classifier.in_features
        self.embedding_dim = in_features
        self.backbone.classifier = nn.Linear(in_features, cfg.num_classes)

        if cfg.freeze_early_layers:
            self.freeze_early_layers(unfreeze_last_denseblock=cfg.unfreeze_last_denseblock)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, l2_normalize: bool = True) -> torch.Tensor:
        features = self.backbone.features(x)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        if l2_normalize:
            out = F.normalize(out, dim=1)
        return out

    def freeze_early_layers(self, unfreeze_last_denseblock: bool = True) -> None:
        for p in self.backbone.features.parameters():
            p.requires_grad = False

        if unfreeze_last_denseblock:
            for name, module in self.backbone.features.named_children():
                if name in {'denseblock4', 'norm5'}:
                    for p in module.parameters():
                        p.requires_grad = True

        for p in self.backbone.classifier.parameters():
            p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
