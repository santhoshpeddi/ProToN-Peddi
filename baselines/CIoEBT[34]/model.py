from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config


def _as_file_url(path: Path) -> str:
    return "file://" + str(path.resolve())


class CaffeNetCaffe2TorchEar(nn.Module):
    """CaffeNet loaded from Caffe files, with a trainable final ear-classification head."""

    def __init__(self, num_classes: int):
        super().__init__()
        try:
            from caffemodel2pytorch import caffemodel2pytorch
        except ImportError as exc:
            raise ImportError(
                "caffemodel2pytorch is required for this baseline. Install the package "
                "and provide the Caffe deploy prototxt, caffemodel, and caffe.proto files."
            ) from exc

        deploy_path = Path(config.caffe_deploy_prototxt)
        weights_path = Path(config.caffe_weights)
        caffe_proto_path = Path(config.caffe_proto_local)

        for label, path in {
            "deploy.prototxt": deploy_path,
            "CaffeNet .caffemodel": weights_path,
            "caffe.proto": caffe_proto_path,
        }.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")

        self.backbone = caffemodel2pytorch.Net(
            prototxt=str(deploy_path),
            weights=str(weights_path),
            caffe_proto=_as_file_url(caffe_proto_path),
        )

        for name in ["prob", "fc8"]:
            if hasattr(self.backbone, name):
                delattr(self.backbone, name)

        self.embedding_dim = 4096
        self.head = nn.Linear(self.embedding_dim, num_classes)

        if config.freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.head.parameters():
            parameter.requires_grad = True

    def _forward_backbone(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.backbone(x)
        if not isinstance(out, dict):
            raise RuntimeError("Expected caffemodel2pytorch.Net to return a dict of blobs.")
        return out

    def extract_bnf(self, x: torch.Tensor) -> torch.Tensor:
        out = self._forward_backbone(x)
        if "fc7" in out:
            return out["fc7"]
        if "fc6" in out:
            return out["fc6"]
        last_key = next(reversed(out.keys()))
        return out[last_key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_bnf(x)
        return self.head(features)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)
