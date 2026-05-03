from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class RandomHistEq:
    def __init__(self, p: float = 0.3):
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        r, g, b = img.split()
        return Image.merge(
            "RGB",
            (ImageOps.equalize(r), ImageOps.equalize(g), ImageOps.equalize(b)),
        )


def build_transforms(split: str, image_size: int = 518) -> T.Compose:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if split == "train":
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
            T.RandomPerspective(distortion_scale=0.2, p=0.3),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.2),
            T.RandomGrayscale(p=0.1),
            RandomHistEq(p=0.3),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class EarDataset(Dataset):
    """Class-consistent train/eval split inside the training folder."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 518,
        eval_ratio: float = 0.2,
        seed: int = 42,
        train_folder: str = "train",
    ):
        assert split in {"train", "eval"}
        self.split = split
        random.seed(seed)

        train_root = Path(root) / train_folder
        if not train_root.is_dir():
            raise RuntimeError(f"Missing train directory: {train_root}")

        train_classes = sorted([d.name for d in train_root.iterdir() if d.is_dir()])
        if not train_classes:
            raise RuntimeError(f"No class folders found inside: {train_root}")

        class_to_images = {}
        for cls in train_classes:
            cls_dir = train_root / cls
            imgs = [
                str(p)
                for p in cls_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
            if len(imgs) < 2:
                continue
            random.shuffle(imgs)
            class_to_images[cls] = imgs

        if not class_to_images:
            raise RuntimeError("No usable classes found (need >=2 images per class)")

        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(class_to_images))}
        self.samples = []
        for cls, imgs in class_to_images.items():
            n_eval = max(1, int(len(imgs) * eval_ratio))
            eval_imgs = imgs[:n_eval]
            train_imgs = imgs[n_eval:]
            selected = train_imgs if split == "train" else eval_imgs
            for path in selected:
                self.samples.append((path, self.class_to_idx[cls]))

        self.transform = build_transforms(split=split, image_size=image_size)

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


def create_loaders(
    root: str | Path,
    image_size: int,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    eval_ratio: float = 0.2,
    seed: int = 42,
    train_folder: str = "train",
):
    train_ds = EarDataset(
        root=root,
        split="train",
        image_size=image_size,
        eval_ratio=eval_ratio,
        seed=seed,
        train_folder=train_folder,
    )
    eval_ds = EarDataset(
        root=root,
        split="eval",
        image_size=image_size,
        eval_ratio=eval_ratio,
        seed=seed,
        train_folder=train_folder,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, eval_loader, len(train_ds.class_to_idx)
