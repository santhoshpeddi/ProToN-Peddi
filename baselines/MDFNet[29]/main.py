from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import config
from dataset import create_loaders
from train import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the MDFNet for ear recognition baseline."
    )
    parser.add_argument("--dataset-root", type=str, required=True, help="Root directory containing train/.")
    parser.add_argument("--out-dir", type=str, default=str(config.OUT_DIR), help="Directory for checkpoints and metrics.")
    parser.add_argument("--train-folder", type=str, default=config.TRAIN_FOLDER, help="Class-folder training directory name.")
    parser.add_argument("--use-internal-eval", action="store_true", help="create a class-wise eval split from train/.")
    parser.add_argument("--eval-ratio", type=float, default=config.EVAL_RATIO, help="Per-class eval split ratio when internal eval is used.")
    parser.add_argument("--subset-id", type=int, default=config.MDFNET_SUBSET, choices=sorted(config.SUBSETS), help="MDFNet subset configuration ID from 1 to 6.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size used for feature extraction.")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS, help="Number of DataLoader workers.")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Random seed for deterministic splitting.")
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE[0], help="Square grayscale image size. MDFNet paper uses 96.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, or cuda.")
    parser.add_argument("--svm-c", type=float, default=config.SVM_C, help="LinearSVC regularization C.")
    parser.add_argument("--svm-max-iter", type=int, default=config.SVM_MAX_ITER, help="Maximum LinearSVC iterations.")
    parser.add_argument("--save-features", action="store_true", help="Save extracted train/eval MDFNet features as .npy files.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return device_arg


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    image_size = (int(args.image_size), int(args.image_size))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, eval_loader, class_to_idx = create_loaders(
        root=args.dataset_root,
        train_folder=args.train_folder,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
    )

    summary = train_and_evaluate(
        train_loader=train_loader,
        eval_loader=eval_loader,
        out_dir=out_dir,
        subset_id=args.subset_id,
        device=device,
        svm_c=args.svm_c,
        svm_max_iter=args.svm_max_iter,
        save_features=args.save_features,
        class_to_idx=class_to_idx,
    )

    print("\n[SUMMARY]")
    print(json.dumps({k: v for k, v in summary.items() if k != "class_to_idx"}, indent=2))


if __name__ == "__main__":
    main()
