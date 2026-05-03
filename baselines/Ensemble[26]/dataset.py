from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from config import TrainConfig


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _is_image_file(name: str) -> bool:
    return name.lower().endswith(IMG_EXTS)


def build_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class EarDataset(Dataset):
    """Class-consistent split built from train-folder classes."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 150,
        eval_ratio: float = 0.2,
        seed: int = 42,
        train_folder: str = "train",
    ):
        assert split in {"train", "eval"}
        self.split = split
        rng = random.Random(seed)

        train_root = Path(root) / train_folder
        if not train_root.is_dir():
            raise RuntimeError(f"Missing train directory: {train_root}")

        train_classes = sorted([d.name for d in train_root.iterdir() if d.is_dir()])
        if not train_classes:
            raise RuntimeError(f"No class folders found inside: {train_root}")

        class_to_images: Dict[str, List[Path]] = {}
        for cls in train_classes:
            cls_dir = train_root / cls
            imgs = [p for p in cls_dir.iterdir() if p.is_file() and _is_image_file(p.name)]
            if len(imgs) < 2:
                continue
            imgs = sorted(imgs)
            rng.shuffle(imgs)
            class_to_images[cls] = imgs

        if not class_to_images:
            raise RuntimeError("No usable classes found (need >=2 images per class)")

        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(class_to_images))}
        self.samples: List[Tuple[Path, int]] = []
        for cls, imgs in class_to_images.items():
            n_eval = max(1, int(len(imgs) * eval_ratio))
            eval_imgs = imgs[:n_eval]
            train_imgs = imgs[n_eval:]
            selected = train_imgs if split == "train" else eval_imgs
            for path in selected:
                self.samples.append((path, self.class_to_idx[cls]))

        self.transform = build_transform(image_size)

        print(
            f"[INFO] {split.upper()} | {len(self.samples)} images | "
            f"{len(self.class_to_idx)} classes | train_root={train_root}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


def create_loaders(cfg: TrainConfig):
    train_ds = EarDataset(
        root=cfg.dataset_root,
        split="train",
        image_size=cfg.image_size,
        eval_ratio=cfg.eval_ratio,
        seed=cfg.seed,
        train_folder=cfg.train_folder,
    )

    eval_ds = EarDataset(
        root=cfg.dataset_root,
        split="eval",
        image_size=cfg.image_size,
        eval_ratio=cfg.eval_ratio,
        seed=cfg.seed,
        train_folder=cfg.train_folder,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, eval_loader, len(train_ds.class_to_idx)
