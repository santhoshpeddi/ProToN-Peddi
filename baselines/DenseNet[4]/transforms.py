from __future__ import annotations

import random
from typing import Tuple

from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class RandomTranslatePixels:
    """Translate horizontally or vertically in the range [-30, +30] pixels."""

    def __init__(self, max_px: int = 30, fill: int | Tuple[int, int, int] = 0):
        self.max_px = int(max_px)
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        dx = random.randint(-self.max_px, self.max_px)
        dy = random.randint(-self.max_px, self.max_px)
        return TF.affine(img, angle=0.0, translate=(dx, dy), scale=1.0, shear=(0.0, 0.0), fill=self.fill)


def build_transforms(split: str, image_size: int = 224):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if split == 'train':
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomRotation(degrees=40),
            RandomTranslatePixels(max_px=30),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    if split in {'eval', 'val', 'test'}:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    raise ValueError(f'Unknown split: {split}')
