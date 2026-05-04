from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def _move_images(images: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        images["convnext"].to(device, non_blocking=True),
        images["efficientnet"].to(device, non_blocking=True),
        images["iresnet"].to(device, non_blocking=True),
    )


def train(cfg, model, criterion, train_loader, val_loader) -> dict:
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = None
    if cfg.use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step, gamma=cfg.scheduler_gamma)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with cfg.metrics_csv.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_model_path = cfg.checkpoint_dir / "best_model.pth"
    last_model_path = cfg.checkpoint_dir / "last_model.pth"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        count = 0

        for images, labels, demographics in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", ncols=100):
            x_conv, x_eff, x_ires = _move_images(images, device)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_conv, x_eff, x_ires)
            if cfg.use_demographic_weights:
                loss = criterion(logits, labels, list(demographics))
            else:
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            train_loss += float(loss.item()) * bs
            train_acc += accuracy_from_logits(logits.detach(), labels) * bs
            count += bs

        train_loss /= max(count, 1)
        train_acc /= max(count, 1)

        val_loss: Optional[float] = None
        val_acc: Optional[float] = None
        do_validate = ((epoch % cfg.val_interval) == 0) or (epoch == cfg.epochs)
        if do_validate and len(val_loader.dataset) > 0:
            val_loss, val_acc = evaluate(cfg, model, criterion, val_loader, device=device)

        if scheduler:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        with cfg.metrics_csv.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, val_loss or "", val_acc or "", lr_now])

        if val_loss is not None:
            print(
                f"Epoch [{epoch}/{cfg.epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr_now:.3e}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = float(val_acc)
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc}, best_model_path)
        else:
            print(f"Epoch [{epoch}/{cfg.epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} val=skipped lr={lr_now:.3e}")

    torch.save({"model": model.state_dict(), "epoch": cfg.epochs}, last_model_path)
    if not best_model_path.is_file():
        torch.save({"model": model.state_dict(), "epoch": cfg.epochs}, best_model_path)

    summary = {
        "best_model": str(best_model_path),
        "last_model": str(last_model_path),
        "metrics_csv": str(cfg.metrics_csv),
        "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
        "best_val_acc": best_val_acc,
        "num_classes": cfg.num_classes,
    }
    cfg.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with cfg.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


@torch.no_grad()
def evaluate(cfg, model, criterion, loader, device=None) -> tuple[float, float]:
    device = device or torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    model.eval()
    if hasattr(criterion, "demographic_weights"):
        original_weights = criterion.demographic_weights
        criterion.demographic_weights = None
    else:
        original_weights = None

    total_loss = 0.0
    total_acc = 0.0
    count = 0
    for images, labels, demographics in loader:
        x_conv, x_eff, x_ires = _move_images(images, device)
        labels = labels.to(device, non_blocking=True)
        logits = model(x_conv, x_eff, x_ires)
        try:
            loss = criterion(logits, labels)
        except TypeError:
            loss = criterion(logits, labels, list(demographics))
        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        total_acc += accuracy_from_logits(logits, labels) * bs
        count += bs

    if hasattr(criterion, "demographic_weights"):
        criterion.demographic_weights = original_weights
    return total_loss / max(count, 1), total_acc / max(count, 1)


@torch.no_grad()
def export_embeddings(cfg, model, loader, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    model.eval().to(device)
    embeddings = []
    labels_all = []
    demographics_all = []
    for images, labels, demographics in loader:
        x_conv, x_eff, x_ires = _move_images(images, device)
        emb = model(x_conv, x_eff, x_ires, return_embedding=True)
        embeddings.append(F.normalize(emb, dim=1).cpu())
        labels_all.append(labels.long().cpu())
        demographics_all.extend(list(demographics))
    torch.save({"embeddings": torch.cat(embeddings, dim=0), "labels": torch.cat(labels_all, dim=0), "demographics": demographics_all}, out_path)
