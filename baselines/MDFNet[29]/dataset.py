from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMG_EXTS


def _collect_class_images(root: Path) -> Dict[str, List[str]]:
    if not root.is_dir():
        raise RuntimeError(f"Missing directory: {root}")

    class_to_images: Dict[str, List[str]] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(str(p) for p in class_dir.iterdir() if p.is_file() and _is_image_file(p))
        if images:
            class_to_images[class_dir.name] = images
    return class_to_images


def _split_no_leakage(images: Sequence[str], eval_ratio: float) -> Tuple[List[str], List[str]]:
    """Return train/eval paths for one identity without putting the same image in both splits."""
    images = list(images)
    n = len(images)
    if n < 2:
        return images, []
    n_eval = max(1, int(n * eval_ratio))
    n_eval = min(n_eval, n - 1)
    return images[n_eval:], images[:n_eval]


def build_transform(image_size: Tuple[int, int] = config.IMAGE_SIZE) -> T.Compose:
    """MDFNet works on deterministic grayscale 96x96 ear-print images."""
    return T.Compose([
        T.Resize(tuple(image_size)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),  # [0,1], shape (1,H,W)
    ])


class EarSplitDataset(Dataset):
    """
    Class-consistent train/eval dataset built from class folders under the train folder.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        eval_ratio: float = config.EVAL_RATIO,
        seed: int = config.SEED,
        train_folder: str = config.TRAIN_FOLDER,
        image_size: Tuple[int, int] = config.IMAGE_SIZE,
        return_path: bool = False,
    ):
        if split not in {"train", "eval"}:
            raise ValueError("split must be 'train' or 'eval'")
        self.split = split
        self.return_path = return_path
        rng = random.Random(seed)

        train_root = Path(root) / train_folder
        class_to_images = _collect_class_images(train_root)
        if not class_to_images:
            raise RuntimeError(f"No usable class folders found inside: {train_root}")

        shuffled: Dict[str, List[str]] = {}
        for class_name, paths in class_to_images.items():
            paths = list(paths)
            rng.shuffle(paths)
            shuffled[class_name] = paths

        self.class_to_idx = {name: idx for idx, name in enumerate(sorted(shuffled))}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        self.samples: List[Tuple[str, int]] = []
        classes_without_eval = 0
        for class_name, paths in shuffled.items():
            train_paths, eval_paths = _split_no_leakage(paths, eval_ratio)
            selected = train_paths if split == "train" else eval_paths
            if split == "eval" and not selected:
                classes_without_eval += 1
            for path in selected:
                self.samples.append((path, self.class_to_idx[class_name]))

        if not self.samples:
            raise RuntimeError(
                f"No samples for split={split!r}. For eval, this can happen when classes contain fewer than two images."
            )

        if split == "eval" and classes_without_eval > 0:
            print(f"[WARN] {classes_without_eval} classes had fewer than two images and contribute no eval samples.")

        self.transform = build_transform(image_size)
        print(
            f"[INFO] {split.upper()} | {len(self.samples)} images | "
            f"{len(self.class_to_idx)} classes | train_root={train_root}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        if self.return_path:
            return image, label, path
        return image, label


def create_loaders(
    root: str | Path,
    train_folder: str = config.TRAIN_FOLDER,
    eval_ratio: float = config.EVAL_RATIO,
    seed: int = config.SEED,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    image_size: Tuple[int, int] = config.IMAGE_SIZE,
):
    train_ds = EarSplitDataset(
        root=root,
        split="train",
        eval_ratio=eval_ratio,
        seed=seed,
        train_folder=train_folder,
        image_size=image_size,
    )

    eval_ds = EarSplitDataset(
        root=root,
        split="eval",
        eval_ratio=eval_ratio,
        seed=seed,
        train_folder=train_folder,
        image_size=image_size,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, eval_loader, train_ds.class_to_idx
