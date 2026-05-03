from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from mdfnet import MDFNet
import config

try:
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
except Exception as exc: 
    raise RuntimeError("scikit-learn is required for MDFNetSVM.") from exc


def loader_to_numpy(loader, device: str = "cpu", aligner=None) -> Tuple[np.ndarray, np.ndarray]:
    """Collect DataLoader batches into grayscale arrays (N,H,W) and labels (N,)."""
    xs, ys = [], []
    torch_device = torch.device(device)
    for batch in loader:
        if len(batch) == 2:
            images, labels = batch
        else:
            images, labels = batch[0], batch[1]
        images = images.to(torch_device, non_blocking=True)
        if aligner is not None:
            images = aligner(images)
        xs.append(images.squeeze(1).detach().cpu().numpy().astype(np.float32))
        ys.append(labels.detach().cpu().numpy().astype(np.int64))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


class MDFNetSVM:
    """MDFNet feature extractor followed by a standardized linear one-vs-rest SVM."""

    def __init__(
        self,
        subset_id: int = config.MDFNET_SUBSET,
        device: str = "cpu",
        svm_c: float = config.SVM_C,
        svm_max_iter: int = config.SVM_MAX_ITER,
    ):
        if subset_id not in config.SUBSETS:
            raise ValueError(f"subset_id must be one of {sorted(config.SUBSETS)}, got {subset_id}")
        hp = config.SUBSETS[subset_id]
        self.subset_id = int(subset_id)
        self.extractor = MDFNet(
            num_filters=hp["num_filters"],
            filter_size=hp["filter_size"],
            block_size=hp["block_size"],
            beta=config.BETA,
            direction_bins=config.DIRECTION_BINS,
            device=device,
        )
        self.device = device
        self.svm_c = float(svm_c)
        self.svm_max_iter = int(svm_max_iter)
        self.classifier = None

    def fit(self, images: np.ndarray, labels: np.ndarray) -> "MDFNetSVM":
        self.extractor.fit(images)
        features = self.extractor.transform(images)
        self.classifier = make_pipeline(
            StandardScaler(),
            LinearSVC(C=self.svm_c, max_iter=self.svm_max_iter),
        )
        self.classifier.fit(features, labels)
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        return self.extractor.transform(images)

    def predict(self, images: np.ndarray) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.classifier.predict(self.transform(images))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "subset_id": self.subset_id,
            "filters": self.extractor.filters_,
            "M": self.extractor.M,
            "p": self.extractor.p,
            "block": self.extractor.block,
            "beta": self.extractor.beta,
            "dir_bins": self.extractor.dir_bins,
            "svm_c": self.svm_c,
            "svm_max_iter": self.svm_max_iter,
            "classifier": self.classifier,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "MDFNetSVM":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        obj = cls(
            subset_id=int(payload["subset_id"]),
            device=device,
            svm_c=float(payload.get("svm_c", config.SVM_C)),
            svm_max_iter=int(payload.get("svm_max_iter", config.SVM_MAX_ITER)),
        )
        obj.extractor.filters_ = payload["filters"]
        obj.classifier = payload["classifier"]
        return obj


def train_and_evaluate(
    train_loader,
    eval_loader,
    out_dir: str | Path,
    subset_id: int,
    device: str = "cpu",
    svm_c: float = config.SVM_C,
    svm_max_iter: int = config.SVM_MAX_ITER,
    save_features: bool = False,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """Train MDFNet+SVM once and save metrics."""
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    metrics_dir = out_dir / "metrics"
    feature_dir = out_dir / "features"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if save_features:
        feature_dir.mkdir(parents=True, exist_ok=True)

    train_images, train_labels = loader_to_numpy(train_loader, device=device)
    eval_images, eval_labels = loader_to_numpy(eval_loader, device=device)

    model = MDFNetSVM(subset_id=subset_id, device=device, svm_c=svm_c, svm_max_iter=svm_max_iter)
    model.fit(train_images, train_labels)

    train_pred = model.predict(train_images)
    eval_pred = model.predict(eval_images)
    train_acc = float(accuracy_score(train_labels, train_pred))
    eval_acc = float(accuracy_score(eval_labels, eval_pred))

    model_path = ckpt_dir / "mdfnet_svm.pkl"
    filters_path = ckpt_dir / "mdfnet_filters.npz"
    model.save(model_path)
    model.extractor.save(str(filters_path))

    if save_features:
        np.save(feature_dir / "train_features.npy", model.transform(train_images))
        np.save(feature_dir / "train_labels.npy", train_labels)
        np.save(feature_dir / "eval_features.npy", model.transform(eval_images))
        np.save(feature_dir / "eval_labels.npy", eval_labels)

    summary = {
        "subset_id": int(subset_id),
        "subset_params": config.SUBSETS[int(subset_id)],
        "train_samples": int(len(train_labels)),
        "eval_samples": int(len(eval_labels)),
        "num_classes": int(len(np.unique(train_labels))),
        "train_accuracy": train_acc,
        "eval_accuracy": eval_acc,
        "svm_c": float(svm_c),
        "svm_max_iter": int(svm_max_iter),
        "model_path": str(model_path),
        "filters_path": str(filters_path),
    }
    if class_to_idx is not None:
        summary["class_to_idx"] = class_to_idx

    with (metrics_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (metrics_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "loss", "accuracy"])
        writer.writeheader()
        writer.writerow({"stage": "train", "loss": 0.0, "accuracy": train_acc})
        writer.writerow({"stage": "eval", "loss": 0.0, "accuracy": eval_acc})

    report = classification_report(eval_labels, eval_pred, output_dict=True, zero_division=0)
    with (metrics_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return summary
