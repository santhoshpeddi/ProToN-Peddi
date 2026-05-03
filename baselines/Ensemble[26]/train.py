from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from config import TrainConfig
from model import build_vgg16_svm, build_vgg19_svm, squared_hinge_loss_one_vs_all


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        logits = model(imgs)
        loss = squared_hinge_loss_one_vs_all(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)

        bs = imgs.size(0)
        total_loss += float(loss.item()) * bs
        correct += int((preds == labels).sum().item())
        total += bs

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        logits = model(imgs)
        loss = squared_hinge_loss_one_vs_all(logits, labels)
        preds = logits.argmax(dim=1)

        bs = imgs.size(0)
        total_loss += float(loss.item()) * bs
        correct += int((preds == labels).sum().item())
        total += bs

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_ensemble_epoch(model1, model2, loader, device, w1: float = 0.5, w2: float = 0.5):
    model1.eval()
    model2.eval()
    denom = (w1 + w2) if (w1 + w2) != 0 else 1.0

    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        p1 = model1.predict_proba(imgs)
        p2 = model2.predict_proba(imgs)
        fused = (w1 * p1 + w2 * p2) / denom
        preds = fused.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += imgs.size(0)

    return correct / max(total, 1)


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_emb, all_lbl = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        emb = model.forward_embedding(imgs)
        all_emb.append(emb.cpu())
        all_lbl.append(labels.to(torch.long).cpu())

    return torch.cat(all_emb, dim=0), torch.cat(all_lbl, dim=0)


def _make_optimizer_for_vgg_svm(model, lr: float, weight_decay: float):
    params = [{"params": list(model.fc1.parameters()) + list(model.fc2.parameters()), "weight_decay": weight_decay}]
    return torch.optim.Adam(params, lr=lr)


def _save_history_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(history_rows: List[Dict], out_dir: Path, tag: str) -> None:
    if not history_rows:
        return

    epochs = [r["epoch"] for r in history_rows]
    tr_loss = [r["train_loss"] for r in history_rows]
    ev_loss = [r["eval_loss"] for r in history_rows]
    tr_acc = [r["train_acc"] for r in history_rows]
    ev_acc = [r["eval_acc"] for r in history_rows]

    plt.figure()
    plt.plot(epochs, tr_loss, label="Train Loss")
    plt.plot(epochs, ev_loss, label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_loss_curve.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, tr_acc, label="Train Acc")
    plt.plot(epochs, ev_acc, label="Eval Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_accuracy_curve.png", dpi=300)
    plt.close()


def train_single_model(cfg: TrainConfig, backbone_name: str, train_loader, eval_loader, num_classes: int, device: torch.device):
    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    cfg.curves_dir.mkdir(parents=True, exist_ok=True)

    name = backbone_name.lower().strip()
    if name == "vgg16":
        model = build_vgg16_svm(
            num_classes=num_classes,
            use_pretrained=cfg.use_pretrained,
            dropout=cfg.dropout,
            weights_path=cfg.vgg16_weights_path,
            allow_torchvision_fallback=cfg.allow_torchvision_fallback,
        ).to(device)
        tag = "vgg16_svm"
    elif name == "vgg19":
        model = build_vgg19_svm(
            num_classes=num_classes,
            use_pretrained=cfg.use_pretrained,
            dropout=cfg.dropout,
            weights_path=cfg.vgg19_weights_path,
            allow_torchvision_fallback=cfg.allow_torchvision_fallback,
        ).to(device)
        tag = "vgg19_svm"
    else:
        raise ValueError("backbone_name must be 'vgg16' or 'vgg19'")

    optimizer = _make_optimizer_for_vgg_svm(model, lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_acc = -1.0
    best_path = cfg.checkpoints_dir / f"{tag}_best.pth"
    history_rows: List[Dict] = []

    epoch_bar = tqdm(range(1, cfg.epochs + 1), desc=f"Training {tag}", ncols=110)
    for epoch in epoch_bar:
        tl, ta = train_epoch(model, train_loader, optimizer, device)
        vl, va = eval_epoch(model, eval_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": float(tl),
            "train_acc": float(ta),
            "eval_loss": float(vl),
            "eval_acc": float(va),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history_rows.append(row)

        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), best_path)

        epoch_bar.set_postfix({"tr_loss": f"{tl:.4f}", "tr_acc": f"{ta:.4f}", "ev_acc": f"{va:.4f}", "best": f"{best_acc:.4f}"})

    model.load_state_dict(torch.load(best_path, map_location=device))
    history_path = cfg.metrics_dir / f"{tag}_history.csv"
    summary_path = cfg.metrics_dir / f"{tag}_summary.json"
    _save_history_csv(history_path, history_rows)
    _plot_curves(history_rows, cfg.curves_dir, tag)

    summary = {
        "model": tag,
        "best_eval_accuracy": float(best_acc),
        "epochs": int(cfg.epochs),
        "batch_size": int(cfg.batch_size),
        "learning_rate": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "dropout": float(cfg.dropout),
        "num_classes": int(num_classes),
        "history_csv": str(history_path),
        "checkpoint": str(best_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return model, history_rows, summary
