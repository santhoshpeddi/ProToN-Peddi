from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from config import DEFAULT_CONFIG, TrainConfig
from dataset import create_loaders
from train import eval_ensemble_epoch, train_single_model


def parse_args() -> argparse.Namespace:
    cfg = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(
        description="Train the Mehta et al. VGG16/VGG19 ensemble ear-recognition baseline."
    )
    parser.add_argument("--dataset-root", type=Path, default=cfg.dataset_root)
    parser.add_argument("--out-dir", type=Path, default=cfg.out_dir)
    parser.add_argument("--weights-dir", type=Path, default=cfg.pretrained_vgg_dir)
    parser.add_argument("--train-folder", type=str, default=cfg.train_folder)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--dropout", type=float, default=cfg.dropout)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--eval-ratio", type=float, default=cfg.eval_ratio)
    parser.add_argument("--image-size", type=int, default=cfg.image_size)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=["vgg16", "vgg19", "ensemble"], default="ensemble")
    parser.add_argument("--w1", type=float, default=cfg.ensemble_weight_vgg16)
    parser.add_argument("--w2", type=float, default=cfg.ensemble_weight_vgg19)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--allow-torchvision-fallback", action="store_true")
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
        pretrained_vgg_dir=args.weights_dir,
        train_folder=args.train_folder,
        image_size=args.image_size,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        ensemble_weight_vgg16=args.w1,
        ensemble_weight_vgg19=args.w2,
        use_pretrained=not args.no_pretrained,
        allow_torchvision_fallback=args.allow_torchvision_fallback,
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
    results = {}

    if args.mode in {"vgg16", "ensemble"}:
        model16, _, summary16 = train_single_model(cfg, "vgg16", train_loader, eval_loader, num_classes, device)
        results["vgg16"] = summary16
    else:
        model16 = None

    if args.mode in {"vgg19", "ensemble"}:
        model19, _, summary19 = train_single_model(cfg, "vgg19", train_loader, eval_loader, num_classes, device)
        results["vgg19"] = summary19
    else:
        model19 = None

    if args.mode == "ensemble":
        assert model16 is not None and model19 is not None
        ensemble_acc = eval_ensemble_epoch(
            model16,
            model19,
            eval_loader,
            device,
            w1=cfg.ensemble_weight_vgg16,
            w2=cfg.ensemble_weight_vgg19,
        )
        ensemble_summary = {
            "model": "vgg16_vgg19_weighted_ensemble",
            "eval_accuracy": float(ensemble_acc),
            "weights": {"vgg16": float(cfg.ensemble_weight_vgg16), "vgg19": float(cfg.ensemble_weight_vgg19)},
        }
        results["ensemble"] = ensemble_summary
        (cfg.metrics_dir / "ensemble_summary.json").write_text(json.dumps(ensemble_summary, indent=2))

    print("[INFO] Run complete.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
