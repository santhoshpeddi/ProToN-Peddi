from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from comb_filter import expand_labels, protect_with_comb_and_expand
from config import config
from dep_ml import DEPML
from model import CaffeNetCaffe2TorchEar


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)

        batch_size = imgs.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def extract_embeddings(model: CaffeNetCaffe2TorchEar, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_embeddings: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        emb = model.extract_bnf(imgs)
        all_embeddings.append(emb.detach().cpu())
        all_labels.append(labels.to(torch.long).cpu())

    X = torch.cat(all_embeddings, dim=0).numpy().astype(np.float64)
    y = torch.cat(all_labels, dim=0).numpy().astype(np.int64)
    return X, y


def train_caffenet_stage(train_loader: DataLoader, eval_loader: DataLoader, num_classes: int, device: torch.device, out_dir: str | Path) -> Tuple[CaffeNetCaffe2TorchEar, List[Dict[str, Any]]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = CaffeNetCaffe2TorchEar(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history: List[Dict[str, Any]] = []
    best_eval_acc = -1.0
    best_path = out_dir / config.best_ckpt
    last_path = out_dir / config.last_ckpt

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        eval_loss, eval_acc = eval_epoch(model, eval_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
        }
        history.append(row)
        print(
            f"[E{epoch:03d}/{config.epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f}"
        )

        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "best_eval_acc": best_eval_acc}, best_path)
        torch.save({"state_dict": model.state_dict(), "epoch": epoch}, last_path)

    with open(out_dir / config.history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    return model, history


def fit_depml_stage(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    eval_labels: np.ndarray,
    out_dir: str | Path,
    apply_comb_protection: bool = True,
    comb_order: int = 6,
    save_arrays: bool = False,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.asarray(train_features, dtype=np.float64)
    y_train = np.asarray(train_labels, dtype=np.int64)
    X_eval = np.asarray(eval_features, dtype=np.float64)
    y_eval = np.asarray(eval_labels, dtype=np.int64)

    if apply_comb_protection:
        X_train = protect_with_comb_and_expand(
            X_train,
            order_L=comb_order,
            aug_factor=config.comb_aug_factor,
            passes=config.comb_multi_pass,
            make_binary=config.earcode_binary,
            seed=config.seed,
        )
        y_train = expand_labels(y_train, config.comb_aug_factor)
        X_eval = protect_with_comb_and_expand(
            X_eval,
            order_L=comb_order,
            aug_factor=1,
            passes=config.comb_multi_pass,
            make_binary=config.earcode_binary,
            seed=config.seed,
        )

    model = DEPML(
        u=config.depml_u,
        l=config.depml_l,
        m_cycles=config.depml_cycles_m,
        eta=config.depml_eta,
        gamma=config.depml_gamma,
        use_eq7=config.depml_use_eq7,
        knn_k=config.depml_knn_k,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_eval)
    eval_acc = float((pred == y_eval).mean())

    np.savez_compressed(
        out_dir / "depml_model_state.npz",
        A=model.A_,
        X=model.X_,
        y=model.y_,
        comb_order=int(comb_order),
        apply_comb_protection=bool(apply_comb_protection),
    )

    if save_arrays:
        np.save(out_dir / "train_features.npy", X_train)
        np.save(out_dir / "train_labels.npy", y_train)
        np.save(out_dir / "eval_features.npy", X_eval)
        np.save(out_dir / "eval_labels.npy", y_eval)

    summary = {
        "depml_eval_acc": eval_acc,
        "train_feature_shape": list(X_train.shape),
        "eval_feature_shape": list(X_eval.shape),
        "comb_protection": bool(apply_comb_protection),
        "comb_order": int(comb_order),
        "comb_aug_factor": int(config.comb_aug_factor) if apply_comb_protection else 1,
        "knn_k": int(config.depml_knn_k),
    }
    with open(out_dir / "depml_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[DEP-ML] eval_acc={eval_acc:.4f}")
    return summary
