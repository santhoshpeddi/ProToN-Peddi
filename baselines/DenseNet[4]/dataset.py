from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from transforms import build_transforms

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def _split_no_leakage(imgs: List[str], eval_ratio: float) -> Tuple[List[str], List[str]]:
    n = len(imgs)
    if n < 2:
        return imgs, []
    n_eval = int(round(n * eval_ratio))
    n_eval = max(1, min(n - 1, n_eval))
    eval_imgs = imgs[:n_eval]
    train_imgs = imgs[n_eval:]
    return train_imgs, eval_imgs


class EarDataset(Dataset):
    def __init__(self, root: str, split: str, image_size: int, eval_ratio: float = 0.2, seed: int = 42, train_folder: str = 'train'):
        assert split in {'train', 'eval'}
        self.split = split
        rng = random.Random(seed)

        train_root = os.path.join(root, train_folder)
        if not os.path.isdir(train_root):
            raise RuntimeError(f'Missing train directory: {train_root}')

        train_classes = sorted([d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))])
        if not train_classes:
            raise RuntimeError(f'No class folders found inside: {train_root}')

        class_to_images: Dict[str, List[str]] = {}
        for cls in train_classes:
            cls_dir = os.path.join(train_root, cls)
            imgs = [os.path.join(cls_dir, fn) for fn in os.listdir(cls_dir) if fn.lower().endswith(IMG_EXTS)]
            if not imgs:
                continue
            rng.shuffle(imgs)
            class_to_images[cls] = imgs

        if not class_to_images:
            raise RuntimeError('No usable classes found (need at least 1 image per class).')

        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(class_to_images))}
        self.samples: List[Tuple[str, int]] = []
        classes_with_no_eval = 0

        for cls, imgs in class_to_images.items():
            train_imgs, eval_imgs = _split_no_leakage(imgs, eval_ratio)
            selected = train_imgs if split == 'train' else eval_imgs
            if not selected:
                classes_with_no_eval += 1
                continue
            for path in selected:
                self.samples.append((path, self.class_to_idx[cls]))

        if not self.samples:
            raise RuntimeError(f"No samples for split='{split}'.")

        if split == 'eval' and classes_with_no_eval > 0:
            print(f'[WARN] {classes_with_no_eval} classes had <2 images, so they contribute 0 samples to EVAL (no leakage).')

        self.transform = build_transforms('train' if split == 'train' else 'eval', image_size=image_size)
        print(f'[INFO] {split.upper()} | {len(self.samples)} images | {len(self.class_to_idx)} classes | train_root={train_root}')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


def create_loaders(root: str, image_size: int, batch_size: int, num_workers: int = 4, pin_memory: bool = True, eval_ratio: float = 0.2, seed: int = 42, train_folder: str = 'train'):
    train_ds = EarDataset(root, 'train', image_size, eval_ratio, seed, train_folder)
    eval_ds = EarDataset(root, 'eval', image_size, eval_ratio, seed, train_folder)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, eval_loader, len(train_ds.class_to_idx)
