from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import config

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _linear_normalize_gray(arr: np.ndarray, nmin: float, nmax: float) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-6:
        return np.full_like(arr, nmin, dtype=np.float32)
    return (arr - mn) / (mx - mn) * (nmax - nmin) + nmin


def load_as_rgb_0_255(path: str | Path) -> Image.Image:
    img_gray = Image.open(path).convert("L")
    arr = np.array(img_gray)

    if config.use_eq12_gray_norm:
        arr = _linear_normalize_gray(arr, config.nmin, config.nmax)
        arr = np.clip(arr, config.nmin, config.nmax).astype(np.uint8)

    img_gray = Image.fromarray(arr.astype(np.uint8), mode="L")
    return Image.merge("RGB", (img_gray, img_gray, img_gray))


def caffe_preprocess(pil_img: Image.Image) -> torch.Tensor:
    """CaffeNet preprocessing: resize, center crop, RGB->BGR, and BGR mean subtraction."""
    img = pil_img.resize((config.caffe_resize, config.caffe_resize))
    crop = config.caffe_crop
    left = (config.caffe_resize - crop) // 2
    top = (config.caffe_resize - crop) // 2
    img = img.crop((left, top, left + crop, top + crop))

    arr = np.array(img, dtype=np.float32)
    arr = arr[:, :, ::-1].copy()  # RGB -> BGR
    arr -= np.array(config.caffe_bgr_mean, dtype=np.float32)
    arr = arr.transpose(2, 0, 1).copy()
    return torch.from_numpy(arr)


def _split_no_leakage(imgs: List[str], eval_ratio: float) -> Tuple[List[str], List[str]]:
    n = len(imgs)
    if n < 2:
        return imgs, []
    n_eval = int(round(n * eval_ratio))
    n_eval = max(1, min(n - 1, n_eval))
    return imgs[n_eval:], imgs[:n_eval]


class EarDataset(Dataset):
    """Class-folder dataset with a class-consistent internal train/eval split."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        eval_ratio: float | None = None,
        seed: int | None = None,
        train_folder: str | None = None,
        return_path: bool = False,
    ):
        assert split in {"train", "eval"}
        self.split = split
        self.return_path = return_path
        eval_ratio = config.eval_ratio if eval_ratio is None else float(eval_ratio)
        seed = config.seed if seed is None else int(seed)
        train_folder = config.train_folder if train_folder is None else train_folder
        rng = random.Random(seed)

        train_root = Path(root) / train_folder
        if not train_root.is_dir():
            raise RuntimeError(f"Missing train directory: {train_root}")

        train_classes = sorted([p.name for p in train_root.iterdir() if p.is_dir()])
        if not train_classes:
            raise RuntimeError(f"No class folders found inside: {train_root}")

        class_to_images: Dict[str, List[str]] = {}
        for cls in train_classes:
            cls_dir = train_root / cls
            imgs = [str(cls_dir / fn) for fn in os.listdir(cls_dir) if fn.lower().endswith(IMG_EXTS)]
            if imgs:
                imgs.sort()
                rng.shuffle(imgs)
                class_to_images[cls] = imgs

        if not class_to_images:
            raise RuntimeError("No usable classes found.")

        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(class_to_images))}
        self.samples: List[Tuple[str, int]] = []
        no_eval = 0

        for cls, imgs in class_to_images.items():
            train_imgs, eval_imgs = _split_no_leakage(imgs, eval_ratio)
            selected = train_imgs if split == "train" else eval_imgs
            if not selected and split == "eval":
                no_eval += 1
                continue
            for path in selected:
                self.samples.append((path, self.class_to_idx[cls]))

        if not self.samples:
            raise RuntimeError(f"No samples for split={split!r}.")

        if split == "eval" and no_eval:
            print(f"[WARN] {no_eval} classes had fewer than two images and contribute no eval sample.")

        print(
            f"[INFO] {split.upper()} | {len(self.samples)} images | "
            f"{len(self.class_to_idx)} classes | train_root={train_root}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img_rgb = load_as_rgb_0_255(path)
        x = caffe_preprocess(img_rgb)
        if self.return_path:
            return x, label, path
        return x, label
