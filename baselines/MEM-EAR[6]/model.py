from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from iresnet import iresnet100


def load_pretrained_weights(model: nn.Module, ckpt_path: str) -> None:
    path = Path(ckpt_path)
    if not ckpt_path or not path.is_file():
        print(f"[WARN] Pretrained checkpoint not found, skipping: {ckpt_path}")
        return

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]

    model_state = model.state_dict()
    filtered_state = {}
    skipped = []

    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key in model_state and model_state[key].shape == value.shape:
            filtered_state[key] = value
        else:
            skipped.append(key)

    model.load_state_dict(filtered_state, strict=False)
    print(f"[INFO] Loaded {len(filtered_state)} keys from: {path}")
    if skipped:
        print(f"[INFO] Skipped {len(skipped)} incompatible keys; examples: {skipped[:5]}")


class ConvNeXtTiny(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = timm.create_model("convnext_tiny", pretrained=False, num_classes=0)
        if config.convnext_pretrained_path:
            load_pretrained_weights(self.backbone, config.convnext_pretrained_path)

        feat_dim = self.backbone.num_features
        self.embedding = nn.Linear(feat_dim, config.embedding_dim)
        self.normalize = config.normalize_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(self.backbone(x))
        return F.normalize(emb, dim=1) if self.normalize else emb


class EfficientNetB3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
        if config.efficientnet_pretrained_path:
            load_pretrained_weights(self.backbone, config.efficientnet_pretrained_path)

        feat_dim = self.backbone.num_features
        self.embedding = nn.Linear(feat_dim, config.embedding_dim)
        self.normalize = config.normalize_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(self.backbone(x))
        return F.normalize(emb, dim=1) if self.normalize else emb


class IResNet100(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = iresnet100(pretrained=False, num_features=config.embedding_dim, dropout=config.dropout)
        if config.iresnet_pretrained_path:
            load_pretrained_weights(self.backbone, config.iresnet_pretrained_path)
        self.normalize = config.normalize_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        return F.normalize(emb, dim=1) if self.normalize else emb


class MEMEar(nn.Module):
    """ConvNeXt-Tiny + EfficientNet-B3 + IResNet-100 feature-fusion model."""

    def __init__(self, config):
        super().__init__()
        if int(config.num_classes) <= 0:
            raise ValueError("config.num_classes must be set before building MEMEar")

        self.convnext = ConvNeXtTiny(config)
        self.efficientnet = EfficientNetB3(config)
        self.iresnet = IResNet100(config)

        weights = torch.tensor(config.fusion_weights, dtype=torch.float32)
        if weights.numel() != 3:
            raise ValueError("fusion_weights must contain exactly three values")
        self.register_buffer("fusion_weights", weights)

        self.dropout = nn.Dropout(p=config.dropout)
        self.classifier = nn.Linear(config.embedding_dim, config.num_classes)

    def forward(self, x_conv, x_eff, x_ires, return_embedding: bool = False):
        e1 = self.convnext(x_conv)
        e2 = self.efficientnet(x_eff)
        e3 = self.iresnet(x_ires)

        weights = self.fusion_weights / self.fusion_weights.sum().clamp_min(1e-12)
        fused = weights[0] * e1 + weights[1] * e2 + weights[2] * e3
        if return_embedding:
            return fused
        return self.classifier(self.dropout(fused))
