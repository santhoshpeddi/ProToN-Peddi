from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from config import DEFAULT_CONFIG, ShapeAutoencoderConfig
from train import train_single_model


def parse_args() -> argparse.Namespace:
    cfg = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(
        description='Train the Pal et al. shape-focused autoencoder ear-recognition baseline.'
    )
    parser.add_argument('--dataset-root', type=Path, default=cfg.dataset_root)
    parser.add_argument('--out-dir', type=Path, default=cfg.out_dir)
    parser.add_argument('--train-folder', type=str, default=cfg.train_folder)
    parser.add_argument('--epochs', type=int, default=cfg.epochs)
    parser.add_argument('--batch-size', type=int, default=cfg.batch_size)
    parser.add_argument('--lr', type=float, default=cfg.lr)
    parser.add_argument('--eval-ratio', type=float, default=cfg.eval_ratio)
    parser.add_argument('--seed', type=int, default=cfg.seed)
    parser.add_argument('--num-workers', type=int, default=cfg.num_workers)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--disable-augmentations', action='store_true')
    parser.add_argument('--disable-standard-scaler', action='store_true')
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(args: argparse.Namespace) -> ShapeAutoencoderConfig:
    return ShapeAutoencoderConfig(
        dataset_root=args.dataset_root,
        train_folder=args.train_folder,
        out_dir=args.out_dir,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        use_augmentations=not args.disable_augmentations,
        use_standard_scaler=not args.disable_standard_scaler,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    seed_everything(cfg.seed)
    summary = train_single_model(cfg, device=args.device)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
