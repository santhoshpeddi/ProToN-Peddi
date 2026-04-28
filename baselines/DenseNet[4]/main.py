from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG, TrainConfig
from dataset import create_loaders
from model import DenseNet161Ear, ModelConfig


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
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += images.size(0)

    return {
        'loss': total_loss / max(1, total_samples),
        'acc': total_correct / max(1, total_samples),
    }


def train_one_epoch(model: nn.Module, loader, optimizer, criterion, device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += images.size(0)

    return {
        'loss': total_loss / max(1, total_samples),
        'acc': total_correct / max(1, total_samples),
    }


def save_history_csv(history: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['epoch', 'lr', 'train_loss', 'train_acc', 'eval_loss', 'eval_acc'],
        )
        writer.writeheader()
        writer.writerows(history)


def plot_curves(history: List[Dict[str, float]], out_dir: Path) -> None:
    if not history:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row['epoch'] for row in history]
    tr_loss = [row['train_loss'] for row in history]
    ev_loss = [row['eval_loss'] for row in history]
    tr_acc = [row['train_acc'] for row in history]
    ev_acc = [row['eval_acc'] for row in history]

    plt.figure()
    plt.plot(epochs, tr_loss, label='Train Loss')
    plt.plot(epochs, ev_loss, label='Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'loss_curve.png', dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, tr_acc, label='Train Acc')
    plt.plot(epochs, ev_acc, label='Eval Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'acc_curve.png', dpi=300)
    plt.close()


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Cannot interpret as bool: {value}')


def train_model(cfg: TrainConfig, device: torch.device) -> Dict[str, float]:
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    cfg.curves_dir.mkdir(parents=True, exist_ok=True)

    train_loader, eval_loader, num_classes = create_loaders(
        root=str(cfg.dataset_root),
        image_size=cfg.input_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        eval_ratio=cfg.eval_ratio,
        seed=cfg.seed,
        train_folder=cfg.train_folder,
    )

    model_cfg = ModelConfig(
        num_classes=num_classes,
        use_local_pretrained=cfg.use_local_pretrained,
        imagenet_weights_path=cfg.weights_path,
        freeze_early_layers=cfg.freeze_early_layers,
        unfreeze_last_denseblock=cfg.unfreeze_last_denseblock,
    )
    model = DenseNet161Ear(model_cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.trainable_parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    best_eval_acc = -1.0
    best_epoch = -1
    history: List[Dict[str, float]] = []

    best_checkpoint_path = cfg.ckpt_dir / 'best_densenet161.pth'
    last_checkpoint_path = cfg.ckpt_dir / 'last_densenet161.pth'
    history_path = cfg.metrics_dir / 'training_history.csv'
    summary_path = cfg.metrics_dir / 'run_summary.json'

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        eval_metrics = evaluate(model, eval_loader, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']

        row = {
            'epoch': epoch,
            'lr': current_lr,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['acc'],
            'eval_loss': eval_metrics['loss'],
            'eval_acc': eval_metrics['acc'],
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}/{cfg.epochs:03d}] "
            f"lr={current_lr:.6f} | "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} | "
            f"eval_loss={row['eval_loss']:.4f} eval_acc={row['eval_acc']:.4f}"
        )

        if eval_metrics['acc'] > best_eval_acc:
            best_eval_acc = eval_metrics['acc']
            best_epoch = epoch
            torch.save(
                {
                    'epoch': epoch,
                    'best_eval_acc': best_eval_acc,
                    'num_classes': num_classes,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_config': asdict(model_cfg),
                    'train_config': asdict(cfg),
                },
                best_checkpoint_path,
            )

    torch.save(
        {
            'epoch': cfg.epochs,
            'num_classes': num_classes,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': asdict(model_cfg),
            'train_config': asdict(cfg),
        },
        last_checkpoint_path,
    )

    save_history_csv(history, history_path)
    plot_curves(history, cfg.curves_dir)

    summary = {
        'model_name': cfg.model_name,
        'best_epoch': best_epoch,
        'best_eval_acc': best_eval_acc,
        'epochs': cfg.epochs,
        'dataset_root': str(cfg.dataset_root),
        'train_folder': cfg.train_folder,
        'use_local_pretrained': cfg.use_local_pretrained,
        'weights_path': str(cfg.weights_path),
        'freeze_early_layers': cfg.freeze_early_layers,
        'unfreeze_last_denseblock': cfg.unfreeze_last_denseblock,
        'best_checkpoint': str(best_checkpoint_path),
        'last_checkpoint': str(last_checkpoint_path),
        'history_csv': str(history_path),
        'loss_curve': str(cfg.curves_dir / 'loss_curve.png'),
        'acc_curve': str(cfg.curves_dir / 'acc_curve.png'),
    }
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f'[INFO] Finished DenseNet161 run | best_eval_acc={best_eval_acc:.4f} @ epoch {best_epoch}')
    print(f'[INFO] Best checkpoint: {best_checkpoint_path}')
    print(f'[INFO] Last checkpoint: {last_checkpoint_path}')
    print(f'[INFO] Metrics:         {history_path}')
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train the El-Naggar and Bourlai thermal-ear DenseNet161 baseline.'
    )
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_CONFIG.dataset_root)
    parser.add_argument('--train-folder', type=str, default=DEFAULT_CONFIG.train_folder)
    parser.add_argument('--input-size', type=int, default=DEFAULT_CONFIG.input_size)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument('--num-workers', type=int, default=DEFAULT_CONFIG.num_workers)
    parser.add_argument('--eval-ratio', type=float, default=DEFAULT_CONFIG.eval_ratio)
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG.epochs)
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG.lr)
    parser.add_argument('--momentum', type=float, default=DEFAULT_CONFIG.momentum)
    parser.add_argument('--weight-decay', type=float, default=DEFAULT_CONFIG.weight_decay)
    parser.add_argument('--weights-path', type=Path, default=DEFAULT_CONFIG.weights_path)
    parser.add_argument('--use-local-pretrained', action='store_true')
    parser.add_argument('--freeze-early-layers', type=str2bool, default=DEFAULT_CONFIG.freeze_early_layers)
    parser.add_argument('--unfreeze-last-denseblock', type=str2bool, default=DEFAULT_CONFIG.unfreeze_last_denseblock)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_CONFIG.out_dir)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
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
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        weights_path=args.weights_path,
        use_local_pretrained=args.use_local_pretrained,
        freeze_early_layers=args.freeze_early_layers,
        unfreeze_last_denseblock=args.unfreeze_last_denseblock,
        out_dir=args.out_dir,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config_from_args(args)
    set_seed(cfg.seed)

    device = torch.device(args.device)
    print(f'[INFO] Using device: {device}')
    print(f'[INFO] Dataset root: {cfg.dataset_root}')
    print(f'[INFO] Output dir:   {cfg.out_dir}')
    train_model(cfg, device)


if __name__ == '__main__':
    main()
