from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset

from config import ShapeAutoencoderConfig
from features import (
    ExtractionParams,
    aug_contrast,
    aug_gamma,
    aug_random_noise,
    extract_distance_features,
    extraction_params_from_config,
)


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def _split_no_leakage(imgs: List[str], eval_ratio: float) -> Tuple[List[str], List[str]]:
    n = len(imgs)
    if n < 2:
        return imgs, []
    n_eval = max(1, int(round(n * eval_ratio)))
    n_eval = min(n_eval, n - 1)
    eval_imgs = imgs[:n_eval]
    train_imgs = imgs[n_eval:]
    return train_imgs, eval_imgs


def _read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return img


def _variants(img_bgr: np.ndarray, cfg: ShapeAutoencoderConfig) -> List[Tuple[str, np.ndarray]]:
    variants = [('orig', img_bgr)]
    if not cfg.use_augmentations:
        return variants
    if cfg.aug_random_noise:
        variants.append(('noise', aug_random_noise(img_bgr, sigma=12.0)))
    if cfg.aug_contrast:
        variants.append(('contrast', aug_contrast(img_bgr, alpha=1.35, beta=0.0)))
    if cfg.aug_gamma:
        variants.append(('gamma', aug_gamma(img_bgr, gamma=1.35)))
    return variants


class EarDataset(Dataset):
    def __init__(
        self,
        cfg: ShapeAutoencoderConfig,
        split: str,
        scaler: Optional[StandardScaler] = None,
        label_encoder: Optional[LabelEncoder] = None,
        fit_scaler_on_train: bool = False,
        save_feature_csv: Optional[str] = None,
    ):
        assert split in {'train', 'eval'}
        self.cfg = cfg
        self.split = split
        rng = random.Random(cfg.seed)

        train_root = cfg.dataset_root / cfg.train_folder
        if not train_root.is_dir():
            raise RuntimeError(f'Missing train directory: {train_root}')

        train_classes = sorted([d.name for d in train_root.iterdir() if d.is_dir()])
        if not train_classes:
            raise RuntimeError(f'No class folders found inside: {train_root}')

        class_to_images: Dict[str, List[str]] = {}
        for cls in train_classes:
            cls_dir = train_root / cls
            imgs = [str(p) for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
            if not imgs:
                continue
            rng.shuffle(imgs)
            class_to_images[cls] = imgs

        if not class_to_images:
            raise RuntimeError('No usable classes found (need at least 1 image per class)')

        self.class_names = sorted(class_to_images.keys())
        if label_encoder is None:
            le = LabelEncoder()
            le.fit(self.class_names)
            self.label_encoder = le
        else:
            self.label_encoder = label_encoder

        raw_samples: List[Tuple[str, str, str]] = []
        classes_with_no_eval = 0
        for cls, imgs in class_to_images.items():
            train_imgs, eval_imgs = _split_no_leakage(imgs, cfg.eval_ratio)
            selected = train_imgs if split == 'train' else eval_imgs
            if len(selected) == 0:
                if split == 'eval':
                    classes_with_no_eval += 1
                continue
            for path in selected:
                img_bgr = _read_bgr(path)
                for aug_name, _variant in _variants(img_bgr, cfg):
                    raw_samples.append((path, cls, aug_name))

        if not raw_samples:
            raise RuntimeError(
                f"No samples for split='{split}'. If many classes have <2 images, eval can be empty by design."
            )

        if split == 'eval' and classes_with_no_eval > 0:
            print(f'[WARN] {classes_with_no_eval} classes had <2 images -> 0 eval samples (no leakage).')

        params: ExtractionParams = extraction_params_from_config(cfg)
        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        meta: List[Tuple[str, str, str]] = []

        for path, cls, aug_name in raw_samples:
            img_bgr = _read_bgr(path)
            if aug_name == 'noise':
                img_bgr = aug_random_noise(img_bgr, sigma=12.0)
            elif aug_name == 'contrast':
                img_bgr = aug_contrast(img_bgr, alpha=1.35, beta=0.0)
            elif aug_name == 'gamma':
                img_bgr = aug_gamma(img_bgr, gamma=1.35)

            feats, _ = extract_distance_features(img_bgr, params=params, debug=False)
            X_list.append(feats)
            y_list.append(int(self.label_encoder.transform([cls])[0]))
            meta.append((path, cls, aug_name))

        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)

        self.scaler = scaler if scaler is not None else StandardScaler()
        if cfg.use_standard_scaler:
            if fit_scaler_on_train:
                self.scaler.fit(X)
            X = self.scaler.transform(X).astype(np.float32)

        if save_feature_csv is not None:
            os.makedirs(os.path.dirname(save_feature_csv) or '.', exist_ok=True)
            df = pd.DataFrame(X, columns=[f'f{i+1}' for i in range(cfg.num_feature_points)])
            df['label_int'] = y
            df['label_name'] = [m[1] for m in meta]
            df['source'] = [os.path.basename(m[0]) for m in meta]
            df['aug'] = [m[2] for m in meta]
            df.to_csv(save_feature_csv, index=False)

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.meta = meta

        print(
            f'[INFO] {split.upper()} | {len(self.X)} samples | {len(self.class_names)} classes | train_root={train_root}'
        )

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        path, cls, aug = self.meta[idx]
        return self.X[idx], self.y[idx], cls, path, aug


def save_train_artifacts(cfg: ShapeAutoencoderConfig, label_encoder: LabelEncoder, scaler: StandardScaler) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.train_classes_path.write_text(json.dumps({'classes': list(label_encoder.classes_)}, indent=2), encoding='utf-8')
    joblib.dump(scaler, cfg.scaler_path)
    joblib.dump(label_encoder, cfg.label_encoder_path)
