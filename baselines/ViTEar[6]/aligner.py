from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pytorch_stacked_hourglass.models.posenet import PoseNet
from pytorch_stacked_hourglass.utils.group import HeatmapParser


class HourglassAligner(nn.Module):
    """Frozen two-stack hourglass aligner."""

    def __init__(self, checkpoint_path: str | Path, device: torch.device, out_size: int = 518):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing hourglass checkpoint: {checkpoint_path}")

        self.device = device
        self.out_size = int(out_size)
        self.model = PoseNet(nstack=2, inp_dim=256, oup_dim=16)

        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device)
        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.parser = HeatmapParser()

    def _crop_and_align(self, img: torch.Tensor, coords, margin: float = 0.2) -> torch.Tensor:
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()

        _, height, width = img.shape
        scale_y = height / 64.0
        scale_x = width / 64.0

        ys = coords[:, 0] * scale_y
        xs = coords[:, 1] * scale_x

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        h = y_max - y_min
        w = x_max - x_min

        y_min -= margin * h
        y_max += margin * h
        x_min -= margin * w
        x_max += margin * w

        y_min = max(0, int(y_min))
        y_max = min(height, int(y_max))
        x_min = max(0, int(x_min))
        x_max = min(width, int(x_max))

        cropped = img[:, y_min:y_max, x_min:x_max]
        return torch.nn.functional.interpolate(
            cropped.unsqueeze(0),
            size=(self.out_size, self.out_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        aligned_imgs = []

        for i in range(imgs.size(0)):
            img = imgs[i]
            if img.shape[-2:] != (256, 256):
                img_resized = torch.nn.functional.interpolate(
                    img.unsqueeze(0),
                    size=(256, 256),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            else:
                img_resized = img

            inp = img_resized.permute(1, 2, 0).unsqueeze(0).to(self.device)
            preds = self.model(inp)
            heatmaps = preds[:, -1]
            parsed = self.parser.parse(heatmaps)
            coords = parsed[0][0][:, :2]
            aligned_imgs.append(self._crop_and_align(img, coords))

        return torch.stack(aligned_imgs, dim=0)
