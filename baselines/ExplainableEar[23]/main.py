from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from config import DEFAULT_CONFIG, TrainConfig
from dataset import create_loaders
from model import ResNetClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return {"loss": avg_loss, "acc": avg_acc}


def train_one_epoch(model: nn.Module, loader, optimizer, criterion, device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return {"loss": avg_loss, "acc": avg_acc}


def save_history_csv(history: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "lr", "train_loss", "train_acc", "eval_loss", "eval_acc"],
        )
        writer.writeheader()
        writer.writerows(history)


def train_variant(cfg: TrainConfig, variant: str, device: torch.device) -> Dict[str, float]:
    print(f"\n[INFO] Starting run for variant: {variant}")
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)

    train_loader, eval_loader, num_classes = create_loaders(
        root=str(cfg.dataset_root),
        input_size=cfg.input_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        eval_ratio=cfg.eval_ratio,
        seed=cfg.seed,
        train_folder=cfg.train_folder,
    )

    model = ResNetClassifier(
        variant=variant,
        num_classes=num_classes,
        weights_dir=str(cfg.weights_dir),
        use_local_pretrained=cfg.use_local_pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    best_eval_acc = -1.0
    best_epoch = -1
    history: List[Dict[str, float]] = []
    best_checkpoint_path = cfg.ckpt_dir / f"best_{variant}.pth"

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        eval_metrics = evaluate(model, eval_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "eval_loss": eval_metrics["loss"],
            "eval_acc": eval_metrics["acc"],
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}/{cfg.epochs:03d}] "
            f"lr={current_lr:.6f} | "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} | "
            f"eval_loss={row['eval_loss']:.4f} eval_acc={row['eval_acc']:.4f}"
        )

        if eval_metrics["acc"] > best_eval_acc:
            best_eval_acc = eval_metrics["acc"]
            best_epoch = epoch
            torch.save(
                {
                    "variant": variant,
                    "epoch": epoch,
                    "num_classes": num_classes,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_eval_acc": best_eval_acc,
                    "config": asdict(cfg),
                },
                best_checkpoint_path,
            )

        scheduler.step()

    history_path = cfg.metrics_dir / f"{variant}_history.csv"
    summary_path = cfg.metrics_dir / f"{variant}_summary.json"
    save_history_csv(history, history_path)

    summary = {
        "variant": variant,
        "best_epoch": best_epoch,
        "best_eval_acc": best_eval_acc,
        "epochs": cfg.epochs,
        "dataset_root": str(cfg.dataset_root),
        "train_folder": cfg.train_folder,
        "use_local_pretrained": cfg.use_local_pretrained,
        "weights_dir": str(cfg.weights_dir),
        "checkpoint": str(best_checkpoint_path),
        "history_csv": str(history_path),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Finished {variant} | best_eval_acc={best_eval_acc:.4f} @ epoch {best_epoch}")
    print(f"[INFO] Checkpoint: {best_checkpoint_path}")
    print(f"[INFO] Metrics:    {history_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Alshazly et al. (2021) ResNet ear-recognition baseline."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_CONFIG.dataset_root)
    parser.add_argument("--train-folder", type=str, default=DEFAULT_CONFIG.train_folder)
    parser.add_argument("--variant", type=str, default=DEFAULT_CONFIG.default_variant, choices=DEFAULT_CONFIG.variants)
    parser.add_argument("--train-all", action="store_true", help="Train all supported ResNet variants sequentially.")

    parser.add_argument("--input-size", type=int, default=DEFAULT_CONFIG.input_size)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG.num_workers)
    parser.add_argument("--eval-ratio", type=float, default=DEFAULT_CONFIG.eval_ratio)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)

    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG.lr)
    parser.add_argument("--step-size", type=int, default=DEFAULT_CONFIG.step_size)
    parser.add_argument("--gamma", type=float, default=DEFAULT_CONFIG.gamma)
    parser.add_argument("--momentum", type=float, default=DEFAULT_CONFIG.momentum)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG.weight_decay)

    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_CONFIG.weights_dir)
    parser.add_argument("--use-local-pretrained", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CONFIG.out_dir)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> TrainConfig:
    return replace(
        DEFAULT_CONFIG,
        dataset_root=args.dataset_root,
        train_folder=args.train_folder,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        step_size=args.step_size,
        gamma=args.gamma,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        weights_dir=args.weights_dir,
        use_local_pretrained=args.use_local_pretrained,
        out_dir=args.out_dir,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config_from_args(args)
    set_seed(cfg.seed)

    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Dataset root: {cfg.dataset_root}")
    print(f"[INFO] Output dir:   {cfg.out_dir}")

    variants = cfg.variants if args.train_all else [args.variant]
    summaries = [train_variant(cfg, variant, device) for variant in variants]

    combined_summary_path = cfg.metrics_dir / "run_summary.json"
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    with combined_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"\n[INFO] Combined summary saved to: {combined_summary_path}")


if __name__ == "__main__":
    main()
