import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import models
    _HAS_WEIGHTS_ENUM = True
except Exception:
    from torchvision import models  # type: ignore
    _HAS_WEIGHTS_ENUM = False


def _build_resnet_no_download(variant: str) -> nn.Module:
    """
    Build a torchvision ResNet WITHOUT triggering any download.
    """
    variant = variant.lower().strip()
    if variant not in {"resnet34", "resnet50", "resnet101", "resnet152"}:
        raise ValueError(f"Unsupported variant: {variant}")

    if _HAS_WEIGHTS_ENUM:
        base = getattr(models, variant)(weights=None)
    else:
        base = getattr(models, variant)(pretrained=False)
    return base


def _load_local_weights_into_resnet(base: nn.Module, weights_path: str):
    """
    Load state_dict from local file into a ResNet.
    """
    if not os.path.isfile(weights_path):
        raise RuntimeError(f"Missing local weights file: {weights_path}")

    sd = torch.load(weights_path, map_location="cpu")

    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    if isinstance(sd, dict):
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    missing, unexpected = base.load_state_dict(sd, strict=False)

    if len(missing) > 0:
        print(f"[WARN] Missing keys when loading weights ({len(missing)}). Example: {missing[:5]}")
    if len(unexpected) > 0:
        print(f"[WARN] Unexpected keys when loading weights ({len(unexpected)}). Example: {unexpected[:5]}")


class ResNetClassifier(nn.Module):
    """
    ResNet backbone + linear classifier.
    Fine-tuning strategy: ALL layers trainable.
    Loads pretrained weights ONLY from local files.
    """
    def __init__(
        self,
        variant: str,
        num_classes: int,
        weights_dir: str = "./pretrained_weights_resnet_imagenet",
        use_local_pretrained: bool = True,
    ):
        super().__init__()
        variant = variant.lower().strip()
        if variant not in {"resnet34", "resnet50", "resnet101", "resnet152"}:
            raise ValueError(f"Unsupported variant: {variant}")

        base = _build_resnet_no_download(variant)

        if use_local_pretrained:
            weights_path = os.path.join(weights_dir, f"{variant}.pth")
            _load_local_weights_into_resnet(base, weights_path)

        # Replace classifier head with Identity to expose embeddings
        in_features = base.fc.in_features
        base.fc = nn.Identity()

        self.backbone = base
        self.classifier = nn.Linear(in_features, num_classes)
        self.embedding_dim = in_features

        # Fine-tuning: train everything
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        feat = self.backbone(x)         # (B, D)
        logits = self.classifier(feat)  # (B, C)
        if return_embedding:
            return logits, feat
        return logits


class ResNetEnsemble(nn.Module):
    """
    Deep ensemble voting:
      - each network -> logits
      - softmax -> posterior probabilities
      - average probabilities across models
      - argmax(mean_probs) -> final class
    """
    def __init__(self, models_list):
        super().__init__()
        self.models = nn.ModuleList(models_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = []
        for m in self.models:
            logits = m(x)
            probs.append(F.softmax(logits, dim=1))
        return torch.stack(probs, dim=0).mean(dim=0)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)

    @torch.no_grad()
    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Concat embeddings from all models then L2 normalize.
        """
        self.eval()
        embs = []
        for m in self.models:
            logits, feat = m(x, return_embedding=True)
            embs.append(feat)
        emb = torch.cat(embs, dim=1)
        emb = F.normalize(emb, dim=1)
        return emb