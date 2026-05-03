from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import TrainConfig
from model import ViTEar


def _cosface_logits(criterion, embeddings: torch.Tensor) -> torch.Tensor:
    weight = F.normalize(criterion.W, dim=1)
    cosine = F.linear(embeddings, weight)
    return criterion.s * cosine


def train_epoch(model, aligner, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        if aligner is not None:
            imgs = aligner(imgs)

        embeddings = model(imgs)
        loss = criterion(embeddings, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            logits = _cosface_logits(criterion, embeddings)
            preds = logits.argmax(dim=1)

        batch_size = imgs.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += int((preds == labels).sum().item())
        total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_epoch(model, aligner, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        if aligner is not None:
            imgs = aligner(imgs)

        embeddings = model(imgs)
        loss = criterion(embeddings, labels)
        logits = _cosface_logits(criterion, embeddings)
        preds = logits.argmax(dim=1)

        batch_size = imgs.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += int((preds == labels).sum().item())
        total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def extract_embeddings(model, aligner, loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        if aligner is not None:
            imgs = aligner(imgs)
        embeddings = model(imgs)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.to(torch.long).cpu())

    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def _save_history_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(history_rows: List[Dict], curves_dir: Path) -> None:
    if not history_rows:
        return
    curves_dir.mkdir(parents=True, exist_ok=True)

    epochs = [r["epoch"] for r in history_rows]
    train_loss = [r["train_loss"] for r in history_rows]
    train_acc = [r["train_acc"] for r in history_rows]
    eval_loss = [r.get("eval_loss") for r in history_rows]
    eval_acc = [r.get("eval_acc") for r in history_rows]

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    if any(v is not None for v in eval_loss):
        plt.plot(epochs, eval_loss, label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curves_dir / "loss_curve.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_acc, label="Train Acc")
    if any(v is not None for v in eval_acc):
        plt.plot(epochs, eval_acc, label="Eval Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curves_dir / "accuracy_curve.png", dpi=300)
    plt.close()


def train_single_model(
    cfg: TrainConfig,
    train_dl,
    eval_dl,
    aligner,
    criterion,
    device,
    num_classes: int,
):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    cfg.curves_dir.mkdir(parents=True, exist_ok=True)

    model = ViTEar(
        embed_dim=cfg.embed_dim,
        backbone_name=cfg.backbone_name,
        pretrained_backbone=cfg.pretrained_backbone,
        backbone_checkpoint=cfg.backbone_checkpoint,
    ).to(device)
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_score = float("-inf")
    history_rows: List[Dict] = []
    best_path = cfg.ckpt_dir / f"best_{cfg.loss_name}.pth"
    last_path = cfg.ckpt_dir / f"last_{cfg.loss_name}.pth"

    epoch_bar = tqdm(range(1, cfg.epochs + 1), desc=f"Training ViTEar [{cfg.loss_name}]", ncols=100)
    for epoch in epoch_bar:
        train_loss, train_acc = train_epoch(model, aligner, train_dl, criterion, optimizer, device)
        eval_loss, eval_acc = eval_epoch(model, aligner, eval_dl, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "eval_loss": float(eval_loss),
            "eval_acc": float(eval_acc),
            "lr": float(cfg.lr),
        }
        history_rows.append(row)
        epoch_bar.set_postfix({"tl": f"{train_loss:.4f}", "ta": f"{train_acc:.4f}", "el": f"{eval_loss:.4f}", "ea": f"{eval_acc:.4f}"})

        if eval_acc > best_score:
            best_score = eval_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_eval_acc": best_score,
                    "loss_name": cfg.loss_name,
                    "num_classes": num_classes,
                    "embed_dim": cfg.embed_dim,
                    "backbone_name": cfg.backbone_name,
                },
                best_path,
            )

    torch.save(
        {
            "epoch": cfg.epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss_name": cfg.loss_name,
            "num_classes": num_classes,
            "embed_dim": cfg.embed_dim,
            "backbone_name": cfg.backbone_name,
        },
        last_path,
    )

    _save_history_csv(cfg.metrics_dir / "history.csv", history_rows)
    _plot_curves(history_rows, cfg.curves_dir)

    summary = {
        "loss_name": cfg.loss_name,
        "epochs": cfg.epochs,
        "num_classes": num_classes,
        "best_eval_acc": best_score,
        "checkpoint": str(best_path),
    }
    with (cfg.metrics_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return model, history_rows, summary
