from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from config import DEFAULT_CONFIG, TrainConfig
from dataset import create_loaders
from train import train_single_model


def parse_args() -> argparse.Namespace:
    cfg = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(
        description="Train the Chowdhury et al. DenseNet-161 ear-recognition baseline."
    )
    parser.add_argument("--dataset-root", type=Path, default=cfg.dataset_root)
    parser.add_argument("--out-dir", type=Path, default=cfg.out_dir)
    parser.add_argument("--weights-path", type=Path, default=cfg.imagenet_weights_path)
    parser.add_argument("--train-folder", type=str, default=cfg.train_folder)
    parser.add_argument("--test-folder", type=str, default=cfg.test_folder)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--momentum", type=float, default=cfg.momentum)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--eval-ratio", type=float, default=cfg.eval_ratio)
    parser.add_argument("--image-size", type=int, default=cfg.image_size)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-class-weights", action="store_true", default=cfg.use_class_weights)
    parser.add_argument("--disable-test-folder-eval", action="store_true")
    parser.add_argument("--no-local-weights", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        train_folder=args.train_folder,
        test_folder=args.test_folder,
        eval_on_test_folder=not args.disable_test_folder_eval,
        imagenet_weights_path=args.weights_path,
        use_local_imagenet_weights=not args.no_local_weights,
        use_class_weights=args.use_class_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        eval_ratio=args.eval_ratio,
        image_size=args.image_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    seed_everything(cfg.seed)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Dataset root: {cfg.dataset_root}")
    print(f"[INFO] Output dir: {cfg.out_dir}")

    train_loader, eval_loader, num_classes = create_loaders(cfg)
    _, _, summary = train_single_model(cfg, train_loader, eval_loader, num_classes, device)

    print("[INFO] Training complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
