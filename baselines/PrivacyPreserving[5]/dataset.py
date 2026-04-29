from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from config import TrainConfig

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class HorizontalCrop:
    def __init__(self, max_ratio: float = 0.2):
        self.max_ratio = float(max_ratio)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        max_total = int(self.max_ratio * w)
        if max_total <= 0:
            return img
        total = random.randint(0, max_total)
        left = random.randint(0, total)
        right = total - left
        return img.crop((left, 0, w - right, h))


class VerticalCrop:
    def __init__(self, max_ratio: float = 0.2):
        self.max_ratio = float(max_ratio)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        max_total = int(self.max_ratio * h)
        if max_total <= 0:
            return img
        total = random.randint(0, max_total)
        top = random.randint(0, total)
        bottom = total - top
        return img.crop((0, top, w, h - bottom))


class GaussianBlurSigma3:
    def __init__(self, sigma: float = 3.0):
        self.sigma = float(sigma)
        self.kernel = int(2 * math.ceil(3 * self.sigma) + 1)

    def __call__(self, img: Image.Image) -> Image.Image:
        return T.GaussianBlur(kernel_size=self.kernel, sigma=self.sigma)(img)


class HistEqualize:
    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.equalize(img)


class AdaptiveHistEqualize:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tuple(tile_grid_size)
        self._warned = False

    def __call__(self, img: Image.Image) -> Image.Image:
        try:
            import cv2
            im = np.array(img.convert("RGB"))
            lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            l2 = clahe.apply(l)
            lab2 = cv2.merge((l2, a, b))
            rgb2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
            return Image.fromarray(rgb2)
        except Exception:
            pass

        try:
            from skimage import exposure
            im = np.array(img.convert("RGB")).astype(np.float32) / 255.0
            out = np.zeros_like(im)
            for c in range(3):
                out[..., c] = exposure.equalize_adapthist(im[..., c], clip_limit=self.clip_limit)
            out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(out)
        except Exception:
            if not self._warned:
                print("[WARN] Adaptive histogram equalization needs opencv-python or scikit-image. Skipping.")
                self._warned = True
            return img


class ClockwiseRotation30:
    def __call__(self, img: Image.Image) -> Image.Image:
        return TF.rotate(img, angle=-30, interpolation=T.InterpolationMode.BILINEAR, expand=False)


class ShearXYFactor02:
    def __init__(self, factor: float = 0.2):
        self.factor = float(factor)
        self.deg = math.degrees(math.atan(self.factor))

    def __call__(self, img: Image.Image) -> Image.Image:
        sx = self.deg * (1 if random.random() < 0.5 else -1)
        sy = self.deg * (1 if random.random() < 0.5 else -1)
        return TF.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[sx, sy],
            interpolation=T.InterpolationMode.BILINEAR,
            fill=0,
        )


class BrightnessPlus20:
    def __init__(self, delta_255: float = 20.0):
        self.delta = float(delta_255) / 255.0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + self.delta, 0.0, 1.0)


class GaussianNoiseMean0Var02:
    def __init__(self, mean: float = 0.0, var: float = 0.2):
        self.mean = float(mean)
        self.var = float(var)
        self.std = math.sqrt(self.var)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.std + self.mean
        return torch.clamp(x + noise, 0.0, 1.0)


def build_transforms(split: str, cfg: TrainConfig) -> T.Compose:
    if split == "train":
        return T.Compose([
            T.RandomApply([HorizontalCrop(max_ratio=0.2)], p=cfg.p_hcrop),
            T.RandomApply([VerticalCrop(max_ratio=0.2)], p=cfg.p_vcrop),
            T.RandomHorizontalFlip(p=cfg.p_hflip),
            T.RandomApply([GaussianBlurSigma3(3.0)], p=cfg.p_blur),
            T.RandomApply([HistEqualize()], p=cfg.p_histeq),
            T.RandomApply([AdaptiveHistEqualize()], p=cfg.p_adapthist),
            T.RandomApply([ClockwiseRotation30()], p=cfg.p_rot30),
            T.RandomApply([ShearXYFactor02(0.2)], p=cfg.p_shear),
            T.Resize((cfg.image_size, cfg.image_size)),
            T.ToTensor(),
            T.RandomApply([BrightnessPlus20(20.0)], p=cfg.p_bright20),
            T.RandomApply([GaussianNoiseMean0Var02(0.0, 0.2)], p=cfg.p_noise),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if split in {"eval", "test", "val"}:
        return T.Compose([
            T.Resize((cfg.image_size, cfg.image_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    raise ValueError(f"Unknown split: {split}")


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def _split_no_leakage(imgs: Sequence[Path], eval_ratio: float) -> Tuple[List[Path], List[Path]]:
    n = len(imgs)
    if n <= 0:
        return [], []
    if n == 1:
        return list(imgs), []
    n_eval = max(1, int(n * eval_ratio))
    if n_eval >= n:
        n_eval = n - 1
    eval_imgs = list(imgs[:n_eval])
    train_imgs = list(imgs[n_eval:])
    return train_imgs, eval_imgs


class EarDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], split: str, cfg: TrainConfig, class_to_idx: Dict[str, int]):
        self.samples = samples
        self.split = split
        self.class_to_idx = class_to_idx
        self.transform = build_transforms(split, cfg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label


def _collect_class_to_images(root: Path) -> Dict[str, List[Path]]:
    if not root.is_dir():
        raise RuntimeError(f"Missing directory: {root}")
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not classes:
        raise RuntimeError(f"No class folders found inside: {root}")
    class_to_images: Dict[str, List[Path]] = {}
    for cls in classes:
        cls_dir = root / cls
        imgs = sorted([p for p in cls_dir.iterdir() if p.is_file() and _is_image_file(p)])
        if imgs:
            class_to_images[cls] = imgs
    if not class_to_images:
        raise RuntimeError(f"No usable class folders found inside: {root}")
    return class_to_images


def create_loaders(cfg: TrainConfig):
    dataset_root = Path(cfg.dataset_root)
    train_root = dataset_root / cfg.train_folder
    test_root = dataset_root / cfg.test_folder

    train_class_to_images = _collect_class_to_images(train_root)
    class_to_idx = {cls: i for i, cls in enumerate(sorted(train_class_to_images.keys()))}

    rng = random.Random(cfg.seed)
    shuffled_train_images: Dict[str, List[Path]] = {}
    for cls, imgs in train_class_to_images.items():
        copied = list(imgs)
        rng.shuffle(copied)
        shuffled_train_images[cls] = copied

    if cfg.eval_on_test_folder and test_root.is_dir():
        test_class_to_images = _collect_class_to_images(test_root)
        shared_classes = sorted(set(class_to_idx.keys()) & set(test_class_to_images.keys()))
        if not shared_classes:
            raise RuntimeError(
                f"No overlapping class folders found between train={train_root} and test={test_root}"
            )

        train_samples: List[Tuple[Path, int]] = []
        eval_samples: List[Tuple[Path, int]] = []
        filtered_mapping = {cls: i for i, cls in enumerate(shared_classes)}

        for cls in shared_classes:
            for path in shuffled_train_images[cls]:
                train_samples.append((path, filtered_mapping[cls]))
            for path in sorted(test_class_to_images[cls]):
                eval_samples.append((path, filtered_mapping[cls]))

        class_to_idx = filtered_mapping
    else:
        train_samples = []
        eval_samples = []
        for cls, imgs in shuffled_train_images.items():
            train_imgs, eval_imgs = _split_no_leakage(imgs, cfg.eval_ratio)
            for path in train_imgs:
                train_samples.append((path, class_to_idx[cls]))
            for path in eval_imgs:
                eval_samples.append((path, class_to_idx[cls]))

        if not eval_samples:
            raise RuntimeError(
                "No evaluation samples were created. Provide a test folder or use classes with at least 2 images."
            )

    train_ds = EarDataset(train_samples, split="train", cfg=cfg, class_to_idx=class_to_idx)
    eval_ds = EarDataset(eval_samples, split="eval", cfg=cfg, class_to_idx=class_to_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    print(
        f"[INFO] TRAIN | {len(train_ds)} images | {len(class_to_idx)} classes | root={train_root}"
    )
    print(
        f"[INFO] EVAL  | {len(eval_ds)} images | {len(class_to_idx)} classes | source={'test folder' if (cfg.eval_on_test_folder and test_root.is_dir()) else 'internal split'}"
    )

    return train_loader, eval_loader, len(class_to_idx)
