from __future__ import annotations

import csv
import json
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ShapeAutoencoderConfig
from dataset import EarDataset, save_train_artifacts
from model import AutoencoderShapeIncorporated


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for feats, labels, _cls, _path, _aug in loader:
        feats = feats.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        probs = model(feats)
        targets = _one_hot(labels, probs.shape[1]).to(device)
        loss = criterion(torch.clamp(probs, 1e-7, 1 - 1e-7), targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = feats.size(0)
        total_loss += float(loss.item()) * bs
        correct += int((probs.argmax(dim=1) == labels).sum().item())
        total += bs

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for feats, labels, _cls, _path, _aug in loader:
        feats = feats.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        probs = model(feats)
        targets = _one_hot(labels, probs.shape[1]).to(device)
        loss = criterion(torch.clamp(probs, 1e-7, 1 - 1e-7), targets)

        bs = feats.size(0)
        total_loss += float(loss.item()) * bs
        correct += int((probs.argmax(dim=1) == labels).sum().item())
        total += bs

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def extract_embeddings(model: AutoencoderShapeIncorporated, loader: DataLoader, device: torch.device):
    model.eval()
    all_emb, all_lbl, all_cls, all_paths = [], [], [], []
    for feats, labels, cls_names, paths, _aug in loader:
        feats = feats.to(device, non_blocking=True).float()
        emb = model.embed(feats, l2_normalize=True).cpu()
        all_emb.append(emb)
        all_lbl.append(labels.long().cpu())
        all_cls.extend(list(cls_names))
        all_paths.extend(list(paths))
    return torch.cat(all_emb, dim=0), torch.cat(all_lbl, dim=0), all_cls, all_paths


def _save_history_csv(history_rows: List[Dict[str, Any]], out_csv) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history_rows[0].keys())
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history_rows)


def _plot_curves(history_rows: List[Dict[str, Any]], cfg: ShapeAutoencoderConfig) -> None:
    epochs = [r['epoch'] for r in history_rows]
    train_loss = [r['train_loss'] for r in history_rows]
    eval_loss = [r['eval_loss'] for r in history_rows]
    train_acc = [r['train_acc'] for r in history_rows]
    eval_acc = [r['eval_acc'] for r in history_rows]

    plt.figure()
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, eval_loss, label='Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.loss_curve_png, dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, eval_acc, label='Eval Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.acc_curve_png, dpi=300)
    plt.close()


def train_single_model(cfg: ShapeAutoencoderConfig, device: Optional[str] = None) -> Dict[str, Any]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))

    train_ds = EarDataset(cfg=cfg, split='train', fit_scaler_on_train=True, save_feature_csv=str(cfg.train_features_csv))
    eval_ds = EarDataset(
        cfg=cfg,
        split='eval',
        scaler=train_ds.scaler,
        label_encoder=train_ds.label_encoder,
        fit_scaler_on_train=False,
        save_feature_csv=str(cfg.eval_features_csv),
    )

    save_train_artifacts(cfg, train_ds.label_encoder, train_ds.scaler)

    num_classes = len(train_ds.label_encoder.classes_)
    model = AutoencoderShapeIncorporated(num_classes=num_classes, cfg=cfg).to(dev)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    eval_dl = DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    history: List[Dict[str, Any]] = []
    best_eval_acc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        tl, ta = train_epoch(model, train_dl, criterion, optimizer, dev)
        el, ea = eval_epoch(model, eval_dl, criterion, dev)

        row = {
            'epoch': epoch,
            'train_loss': float(tl),
            'train_acc': float(ta),
            'eval_loss': float(el),
            'eval_acc': float(ea),
        }
        history.append(row)
        print(f"[E{epoch:03d}] train_loss={tl:.4f} train_acc={ta:.4f} eval_loss={el:.4f} eval_acc={ea:.4f}")

        if ea > best_eval_acc:
            best_eval_acc = ea
            torch.save(model.state_dict(), cfg.best_model_path)

    torch.save(model.state_dict(), cfg.model_weights_path)
    _save_history_csv(history, cfg.history_csv)
    _plot_curves(history, cfg)

    summary = {
        'model': 'autoencoder_shape_incorporated',
        'num_classes': int(num_classes),
        'best_eval_acc': float(best_eval_acc),
        'final_model': str(cfg.model_weights_path),
        'best_model': str(cfg.best_model_path),
        'history_csv': str(cfg.history_csv),
        'train_features_csv': str(cfg.train_features_csv),
        'eval_features_csv': str(cfg.eval_features_csv),
        'scaler_path': str(cfg.scaler_path),
        'label_encoder_path': str(cfg.label_encoder_path),
        'train_classes_path': str(cfg.train_classes_path),
    }
    cfg.summary_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
