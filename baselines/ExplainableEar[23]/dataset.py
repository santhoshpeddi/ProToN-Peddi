import os
import random
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageFilter

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _is_image_file(p: str) -> bool:
    return os.path.splitext(p.lower())[1] in IMG_EXTS


class ScaleAndPasteToCanvas:
    """
    Scale the image randomly and paste it into a canvas, such that 80% to 100%
    of the canvas area is covered. Background filled with ImageNet mean.
    """
    def __init__(self, canvas_size: int, area_range=(0.80, 1.00)):
        self.canvas_size = int(canvas_size)
        self.area_min = float(area_range[0])
        self.area_max = float(area_range[1])
        self.bg_color = tuple(int(round(m * 255.0)) for m in IMAGENET_MEAN)

    def __call__(self, img: Image.Image) -> Image.Image:
        cw = ch = self.canvas_size

        area = random.uniform(self.area_min, self.area_max)
        lin = float(np.sqrt(area))

        target_w = max(1, int(round(cw * lin)))
        target_h = max(1, int(round(ch * lin)))

        w, h = img.size
        scale = min(target_w / max(1, w), target_h / max(1, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = img.resize((new_w, new_h), resample=Image.BILINEAR)

        canvas = Image.new("RGB", (cw, ch), self.bg_color)
        max_x = cw - new_w
        max_y = ch - new_h
        x = random.randint(0, max(0, max_x))
        y = random.randint(0, max(0, max_y))
        canvas.paste(resized, (x, y))
        return canvas


class RandomAspectPreservingCrop:
    """
    Crop randomly to 90% to 100% of original size, keeping aspect ratio.
    """
    def __init__(self, scale_range=(0.90, 1.00)):
        self.smin = float(scale_range[0])
        self.smax = float(scale_range[1])

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = random.uniform(self.smin, self.smax)
        cw = max(1, int(round(w * s)))
        ch = max(1, int(round(h * s)))

        if cw == w and ch == h:
            return img

        x0 = random.randint(0, max(0, w - cw))
        y0 = random.randint(0, max(0, h - ch))
        return img.crop((x0, y0, x0 + cw, y0 + ch))


class RandomGaussianBlur:
    def __init__(self, p=0.20):
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        if random.random() < 0.5:
            return img.filter(ImageFilter.GaussianBlur(radius=0.8))  
        return img.filter(ImageFilter.GaussianBlur(radius=1.2))      


class AddGaussianNoise:
    """
    Mix the image with Gaussian noise of random amount (on tensor in [0,1] pre-normalize).
    """
    def __init__(self, sigma_range=(0.0, 0.08), p=1.0):
        self.smin = float(sigma_range[0])
        self.smax = float(sigma_range[1])
        self.p = float(p)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.p < 1.0 and random.random() > self.p:
            return x
        sigma = random.uniform(self.smin, self.smax)
        if sigma <= 0:
            return x
        noise = torch.randn_like(x) * sigma
        return torch.clamp(x + noise, 0.0, 1.0)


def build_train_transform(input_size: int) -> T.Compose:
    """
    Paper Ref: Augmentation list:
    - scale & paste into canvas (80%-100% area), background ImageNet mean
    - rotate [-20, 20]
    - shear up to 7.5% width
    - crop 90%-100% keep aspect
    - resize to canvas
    - blur p=0.2 (kernel 3 or 4)
    - gaussian noise random amount
    - brightness [-20%, 20%]
    - contrast [-40%, 40%]
    - saturation [-20%, 20%]
    - hue [-3%, 3%]
    - flip p=0.5
    """
    size = int(input_size)

    rotate_shear = T.RandomAffine(
        degrees=20,
        shear=(-4.3, 4.3),  # arctan(0.075) ~= 4.29 degrees
        interpolation=T.InterpolationMode.BILINEAR,
        fill=tuple(int(round(m * 255.0)) for m in IMAGENET_MEAN),
    )

    color = T.ColorJitter(
        brightness=(0.8, 1.2),
        contrast=(0.6, 1.4),
        saturation=(0.8, 1.2),
        hue=(-0.03, 0.03),
    )

    return T.Compose([
        ScaleAndPasteToCanvas(canvas_size=size, area_range=(0.80, 1.00)),
        rotate_shear,
        RandomAspectPreservingCrop(scale_range=(0.90, 1.00)),
        T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
        RandomGaussianBlur(p=0.20),
        color,
        T.RandomHorizontalFlip(p=0.50),
        T.ToTensor(),
        AddGaussianNoise(sigma_range=(0.0, 0.08), p=1.0),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform(input_size: int) -> T.Compose:
    size = int(input_size)
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class EarTrainEvalDataset(Dataset):
    """
    Train/eval split done INSIDE each class folder of TRAIN folder.
    Uses ONLY classes from train_folder.
    """
    def __init__(
        self,
        root: str,
        split: str,                 
        input_size: int,
        eval_ratio: float = 0.2,
        seed: int = 42,
        train_folder: str = "train",
    ):
        assert split in {"train", "eval"}
        self.split = split

        random.seed(seed)

        train_root = os.path.join(root, train_folder)
        if not os.path.isdir(train_root):
            raise RuntimeError(f"Missing train directory: {train_root}")

        classes = sorted([
            d for d in os.listdir(train_root)
            if os.path.isdir(os.path.join(train_root, d))
        ])
        if len(classes) == 0:
            raise RuntimeError(f"No class folders found in: {train_root}")

        class_to_images: Dict[str, List[str]] = {}
        for cls in classes:
            cls_dir = os.path.join(train_root, cls)
            imgs = [
                os.path.join(cls_dir, fn)
                for fn in os.listdir(cls_dir)
                if _is_image_file(fn)
            ]
            if len(imgs) < 2:
                continue
            imgs = sorted(imgs)
            random.shuffle(imgs)
            class_to_images[cls] = imgs

        if len(class_to_images) == 0:
            raise RuntimeError("No usable classes found (need >=2 images per class)")

        self.class_to_idx = {c: i for i, c in enumerate(sorted(class_to_images.keys()))}

        self.samples: List[Tuple[str, int]] = []
        for cls, imgs in class_to_images.items():
            n_eval = max(1, int(len(imgs) * eval_ratio))
            eval_imgs = imgs[:n_eval]
            train_imgs = imgs[n_eval:] if len(imgs) - n_eval > 0 else imgs[:]

            selected = train_imgs if split == "train" else eval_imgs
            for p in selected:
                self.samples.append((p, self.class_to_idx[cls]))

        self.transform = build_train_transform(input_size) if split == "train" else build_eval_transform(input_size)

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
    root: str,
    input_size: int,
    batch_size: int,
    num_workers: int = 4,
    eval_ratio: float = 0.2,
    seed: int = 42,
    train_folder: str = "train"
):
    train_ds = EarTrainEvalDataset(root, "train", input_size, eval_ratio, seed, train_folder)
    eval_ds  = EarTrainEvalDataset(root, "eval",  input_size, eval_ratio, seed, train_folder)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    num_classes = len(train_ds.class_to_idx)
    return train_loader, eval_loader, num_classes