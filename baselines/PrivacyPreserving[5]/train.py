from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from config import TrainConfig
from model import DenseNet161Ear


def _compute_class_weights_from_loader(train_loader, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, labels in train_loader:
        for y in labels:
            counts[int(y)] += 1.0
    counts = torch.clamp(counts, min=1.0)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return weights.to(device)


def _maybe_reduce_lr_on_plateau(
    prev_train_loss: Optional[float],
    curr_train_loss: float,
    bad_count: int,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> Tuple[int, float]:
    if prev_train_loss is not None and curr_train_loss >= prev_train_loss:
        bad_count += 1
    else:
        bad_count = 0

    if bad_count >= cfg.lr_plateau_patience:
        for pg in optimizer.param_groups:
            pg["lr"] = max(pg["lr"] / cfg.lr_reduce_factor, cfg.min_lr)
        bad_count = 0

    return bad_count, optimizer.param_groups[0]["lr"]


def train_epoch(model, loader, criterion, optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        correct += int((preds == labels).sum().item())
        total += bs

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        correct += int((preds == labels).sum().item())
        total += bs

    return total_loss / max(total, 1), correct / max(total, 1)


def _save_history_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(rows: List[Dict], curves_dir: Path) -> None:
    if not rows:
        return

    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    eval_loss = [r["eval_loss"] for r in rows]
    train_acc = [r["train_acc"] for r in rows]
    eval_acc = [r["eval_acc"] for r in rows]

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, eval_loss, label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curves_dir / "loss_curve.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_acc, label="Train Acc")
    plt.plot(epochs, eval_acc, label="Eval Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curves_dir / "accuracy_curve.png", dpi=300)
    plt.close()


def train_single_model(cfg: TrainConfig, train_loader, eval_loader, num_classes: int, device: torch.device):
    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    cfg.curves_dir.mkdir(parents=True, exist_ok=True)

    model = DenseNet161Ear(cfg=cfg, num_classes=num_classes).to(device)

    if cfg.use_class_weights:
        class_weights = _compute_class_weights_from_loader(train_loader, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    best_metric = -1.0
    history: List[Dict] = []
    prev_train_loss = None
    bad_count = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        eval_loss, eval_acc = eval_epoch(model, eval_loader, criterion, device)

        bad_count, lr_now = _maybe_reduce_lr_on_plateau(prev_train_loss, train_loss, bad_count, optimizer, cfg)
        prev_train_loss = train_loss

        row = {
            "epoch": epoch,
            "lr": float(lr_now),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "eval_loss": float(eval_loss),
            "eval_acc": float(eval_acc),
        }
        history.append(row)

        print(
            f"Epoch [{epoch:03d}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f} "
            f"| lr={lr_now:.3e}"
        )

        selection_metric = eval_acc if len(eval_loader.dataset) > 0 else train_acc
        if selection_metric > best_metric:
            best_metric = selection_metric
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "config": cfg.__dict__,
                },
                cfg.checkpoints_dir / "best_model.pt",
            )

    torch.save(
        {
            "epoch": cfg.epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg.__dict__,
        },
        cfg.checkpoints_dir / "last_model.pt",
    )

    _save_history_csv(cfg.metrics_dir / "history.csv", history)
    _plot_curves(history, cfg.curves_dir)

    summary = {
        "epochs": cfg.epochs,
        "best_metric": best_metric,
        "final_train_loss": history[-1]["train_loss"],
        "final_train_acc": history[-1]["train_acc"],
        "final_eval_loss": history[-1]["eval_loss"],
        "final_eval_acc": history[-1]["eval_acc"],
        "use_class_weights": cfg.use_class_weights,
        "use_local_imagenet_weights": cfg.use_local_imagenet_weights,
    }
    with (cfg.metrics_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return model, history, summary
