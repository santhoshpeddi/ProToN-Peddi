from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import Config
from loss import WeightedCrossEntropyLoss
from model import MEMEar
from train import train, export_embeddings
from utils import (
    build_dataloaders,
    build_eval_transforms,
    build_train_transforms,
    compute_demographic_weights,
    generate_csvs,
    split_train_val_imagewise,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UERC 2023 participant-style MEM-Ear fusion baseline")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset root containing train/ folder")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/memear_fusion"), help="Output directory")
    parser.add_argument("--train-folder", default="train", help="Training folder name inside dataset root")

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--fusion-weights", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Weights for ConvNeXt, EfficientNet, IResNet branches")

    parser.add_argument("--convnext-weights", default="", help="Local ConvNeXt-Tiny checkpoint")
    parser.add_argument("--efficientnet-weights", default="", help="Local EfficientNet-B3 checkpoint")
    parser.add_argument("--iresnet-weights", default="", help="Local IResNet-100 checkpoint")

    parser.add_argument("--use-demographic-weights", action="store_true", help="Use inverse-frequency demographic sample weighting")
    parser.add_argument("--demographics-csv", type=Path, default=None, help="CSV containing cid, gender, ethnicity columns")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    cfg = Config()
    cfg.dataset_root = args.dataset_root
    cfg.output_dir = args.out_dir
    cfg.train_folder = args.train_folder
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.validation_ratio = args.val_ratio
    cfg.val_interval = args.val_interval
    cfg.num_workers = args.num_workers
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.embedding_dim = args.embedding_dim
    cfg.dropout = args.dropout
    cfg.fusion_weights = list(args.fusion_weights)
    cfg.convnext_pretrained_path = args.convnext_weights
    cfg.efficientnet_pretrained_path = args.efficientnet_weights
    cfg.iresnet_pretrained_path = args.iresnet_weights
    cfg.use_demographic_weights = args.use_demographic_weights
    cfg.demographics_csv = args.demographics_csv

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = generate_csvs(cfg, allow_identity_overlap=args.allow_identity_overlap)
    train_split_csv, val_split_csv, id2label = split_train_val_imagewise(
        train_csv=train_csv,
        out_dir=cfg.csv_dir,
        val_ratio=cfg.validation_ratio,
        seed=cfg.split_seed,
    )
    cfg.num_classes = len(id2label)

    train_transforms = build_train_transforms(cfg)
    eval_transforms = build_eval_transforms(cfg)
    train_loader, val_loader= build_dataloaders(
        cfg, train_transforms, eval_transforms, train_split_csv, val_split_csv
    )

    model = MEMEar(cfg)
    if cfg.use_demographic_weights:
        demographic_weights = compute_demographic_weights(train_split_csv)
        criterion = WeightedCrossEntropyLoss(demographic_weights=demographic_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    summary = train(cfg, model, criterion, train_loader, val_loader)
    print(f"[INFO] Training summary: {summary}")


if __name__ == "__main__":
    main()
