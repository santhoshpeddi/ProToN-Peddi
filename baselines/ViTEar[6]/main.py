from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from aligner import HourglassAligner
from config import DEFAULT_CONFIG, TrainConfig
from dataset import create_loaders
from losses import CosFace, ElasticCosFace, ElasticCosFacePlus
from train import train_single_model


def parse_args() -> argparse.Namespace:
    cfg = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(
        description="Train the UERC 2023 participant-style ViT ear-recognition baseline."
    )
    parser.add_argument("--dataset-root", type=Path, default=cfg.dataset_root)
    parser.add_argument("--train-folder", type=str, default=cfg.train_folder)
    parser.add_argument("--out-dir", type=Path, default=cfg.out_dir)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=cfg.eval_batch_size)
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    parser.add_argument("--eval-ratio", type=float, default=cfg.eval_ratio)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--image-size", type=int, default=cfg.image_size)
    parser.add_argument("--embed-dim", type=int, default=cfg.embed_dim)
    parser.add_argument("--backbone-name", type=str, default=cfg.backbone_name)
    parser.add_argument("--backbone-checkpoint", type=Path, default=cfg.backbone_checkpoint)
    parser.add_argument("--pretrained-backbone", dest="pretrained_backbone", action="store_true")
    parser.add_argument("--no-pretrained-backbone", dest="pretrained_backbone", action="store_false")
    parser.set_defaults(pretrained_backbone=cfg.pretrained_backbone)
    parser.add_argument("--use-aligner", action="store_true", default=cfg.use_aligner)
    parser.add_argument("--hg-ckpt", type=Path, default=cfg.hg_ckpt)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument(
        "--loss-name",
        type=str,
        choices=["cosface", "elastic_cosface", "elastic_cosface_plus"],
        default=cfg.loss_name,
    )
    parser.add_argument("--cosface-scale", type=float, default=cfg.cosface_scale)
    parser.add_argument("--cosface-margin", type=float, default=cfg.cosface_margin)
    parser.add_argument("--elastic-std", type=float, default=cfg.elastic_std)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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
        train_folder=args.train_folder,
        out_dir=args.out_dir,
        hg_ckpt=args.hg_ckpt,
        use_aligner=args.use_aligner,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        image_size=args.image_size,
        embed_dim=args.embed_dim,
        backbone_name=args.backbone_name,
        pretrained_backbone=args.pretrained_backbone,
        backbone_checkpoint=args.backbone_checkpoint,
        lr=args.lr,
        loss_name=args.loss_name,
        cosface_scale=args.cosface_scale,
        cosface_margin=args.cosface_margin,
        elastic_std=args.elastic_std,
    )


def build_loss(cfg: TrainConfig, num_classes: int):
    kwargs = {"s": cfg.cosface_scale, "m": cfg.cosface_margin}
    if cfg.loss_name == "cosface":
        return CosFace(cfg.embed_dim, num_classes, **kwargs)
    if cfg.loss_name == "elastic_cosface":
        return ElasticCosFace(cfg.embed_dim, num_classes, std=cfg.elastic_std, **kwargs)
    if cfg.loss_name == "elastic_cosface_plus":
        return ElasticCosFacePlus(cfg.embed_dim, num_classes, std=cfg.elastic_std, **kwargs)
    raise ValueError(f"Unsupported loss: {cfg.loss_name}")


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    seed_everything(cfg.seed)

    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Dataset root: {cfg.dataset_root}")
    print(f"[INFO] Output dir: {cfg.out_dir}")

    train_loader, eval_loader, num_classes = create_loaders(
        root=cfg.dataset_root,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        eval_ratio=cfg.eval_ratio,
        seed=cfg.seed,
        train_folder=cfg.train_folder,
    )

    aligner = None
    if cfg.use_aligner:
        if cfg.hg_ckpt is None:
            raise ValueError("--use-aligner requires --hg-ckpt")
        aligner = HourglassAligner(cfg.hg_ckpt, device=device, out_size=cfg.image_size)

    criterion = build_loss(cfg, num_classes)
    _, _, summary = train_single_model(cfg, train_loader, eval_loader, aligner, criterion, device, num_classes)
    print("[INFO] Training complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
