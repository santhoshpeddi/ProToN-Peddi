from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import models
    from torchvision.models import VGG16_Weights, VGG19_Weights
    _HAS_WEIGHTS_ENUM = True
except Exception:
    from torchvision import models  # type: ignore
    VGG16_Weights = None  # type: ignore
    VGG19_Weights = None  # type: ignore
    _HAS_WEIGHTS_ENUM = False


class VGGSVM(nn.Module):
    """Frozen VGG feature extractor with classifier head."""

    def __init__(self, backbone: nn.Module, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        in_dim = 512 * 4 * 4
        self.fc1 = nn.Linear(in_dim, 128)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward_embedding(x)
        emb = self.drop(emb)
        return self.fc2(emb)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return F.softmax(self.forward(x), dim=1)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
    return ckpt


def _try_load_local_vgg_weights(backbone: nn.Module, local_path: Path) -> bool:
    if not local_path.is_file():
        return False

    try:
        ckpt = torch.load(local_path, map_location="cpu")
        state_dict = _extract_state_dict(ckpt)
        if not isinstance(state_dict, dict):
            return False
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded local weights: {local_path}")
        if missing:
            print(f"[WARN] Missing keys while loading {local_path.name}: {len(missing)}")
        if unexpected:
            print(f"[WARN] Unexpected keys while loading {local_path.name}: {len(unexpected)}")
        return True
    except Exception as exc:
        print(f"[WARN] Failed to load local weights from {local_path}: {exc}")
        return False


def _make_vgg16_backbone(use_pretrained: bool, weights_path: Path | None = None, allow_torchvision_fallback: bool = False) -> nn.Module:
    if not use_pretrained:
        return models.vgg16(weights=None) if _HAS_WEIGHTS_ENUM else models.vgg16(pretrained=False)

    backbone = models.vgg16(weights=None) if _HAS_WEIGHTS_ENUM else models.vgg16(pretrained=False)
    if weights_path is not None and _try_load_local_vgg_weights(backbone, weights_path):
        return backbone

    if allow_torchvision_fallback:
        print("[INFO] Local VGG16 weights unavailable. Falling back to torchvision ImageNet weights.")
        if _HAS_WEIGHTS_ENUM:
            return models.vgg16(weights=VGG16_Weights.DEFAULT)
        return models.vgg16(pretrained=True)  # type: ignore[arg-type]

    print("[WARN] Proceeding without pretrained VGG16 weights because no local weights were found.")
    return backbone


def _make_vgg19_backbone(use_pretrained: bool, weights_path: Path | None = None, allow_torchvision_fallback: bool = False) -> nn.Module:
    if not use_pretrained:
        return models.vgg19(weights=None) if _HAS_WEIGHTS_ENUM else models.vgg19(pretrained=False)

    backbone = models.vgg19(weights=None) if _HAS_WEIGHTS_ENUM else models.vgg19(pretrained=False)
    if weights_path is not None and _try_load_local_vgg_weights(backbone, weights_path):
        return backbone

    if allow_torchvision_fallback:
        print("[INFO] Local VGG19 weights unavailable. Falling back to torchvision ImageNet weights.")
        if _HAS_WEIGHTS_ENUM:
            return models.vgg19(weights=VGG19_Weights.DEFAULT)
        return models.vgg19(pretrained=True)  # type: ignore[arg-type]

    print("[WARN] Proceeding without pretrained VGG19 weights because no local weights were found.")
    return backbone


def build_vgg16_svm(
    num_classes: int,
    use_pretrained: bool = True,
    dropout: float = 0.5,
    weights_path: Path | None = None,
    allow_torchvision_fallback: bool = False,
) -> VGGSVM:
    backbone = _make_vgg16_backbone(use_pretrained, weights_path, allow_torchvision_fallback)
    for parameter in backbone.features.parameters():
        parameter.requires_grad = False
    return VGGSVM(backbone, num_classes=num_classes, dropout=dropout)


def build_vgg19_svm(
    num_classes: int,
    use_pretrained: bool = True,
    dropout: float = 0.5,
    weights_path: Path | None = None,
    allow_torchvision_fallback: bool = False,
) -> VGGSVM:
    backbone = _make_vgg19_backbone(use_pretrained, weights_path, allow_torchvision_fallback)
    for parameter in backbone.features.parameters():
        parameter.requires_grad = False
    return VGGSVM(backbone, num_classes=num_classes, dropout=dropout)


def squared_hinge_loss_one_vs_all(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    y = torch.full_like(logits, -1.0)
    y.scatter_(1, targets.view(-1, 1), 1.0)
    margins = 1.0 - y * logits
    return torch.clamp(margins, min=0.0).pow(2).mean()
