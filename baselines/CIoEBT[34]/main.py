from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import config
from dataset import EarDataset
from model import CaffeNetCaffe2TorchEar
from train import extract_embeddings, fit_depml_stage, train_caffenet_stage


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_orders(values: Iterable[str]) -> List[int]:
    out: List[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    if not out:
        raise ValueError("At least one comb-filter order must be provided.")
    return out


def update_config_from_args(args: argparse.Namespace) -> None:
    config.data_root = Path(args.dataset_root)
    config.out_dir = Path(args.out_dir)
    config.train_folder = args.train_folder
    config.eval_ratio = args.eval_ratio
    config.seed = args.seed

    config.caffe_deploy_prototxt = Path(args.caffe_deploy)
    config.caffe_weights = Path(args.caffe_weights)
    config.caffe_proto_local = Path(args.caffe_proto)

    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.lr = args.lr
    config.weight_decay = args.weight_decay
    config.freeze_backbone = not args.unfreeze_backbone

    config.depml_u = args.depml_u
    config.depml_l = args.depml_l
    config.depml_cycles_m = args.depml_cycles
    config.depml_eta = args.depml_eta
    config.depml_gamma = args.depml_gamma
    config.depml_knn_k = args.knn_k

    config.comb_aug_factor = args.comb_aug_factor
    config.comb_multi_pass = args.comb_passes
    config.earcode_binary = not args.real_valued_earcode


def build_loaders(args: argparse.Namespace):
    train_ds = EarDataset(
        root=config.data_root,
        split="train",
        eval_ratio=config.eval_ratio,
        seed=config.seed,
        train_folder=config.train_folder,
    )
    eval_ds = EarDataset(
        root=config.data_root,
        split="eval",
        eval_ratio=config.eval_ratio,
        seed=config.seed,
        train_folder=config.train_folder,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_ds, eval_ds, train_loader, eval_loader


def load_feature_arrays(args: argparse.Namespace):
    required = [args.train_features, args.train_labels, args.eval_features, args.eval_labels]
    if any(v is None for v in required):
        raise RuntimeError(
            "Feature-array mode requires --train-features, --train-labels, --eval-features, and --eval-labels."
        )
    return (
        np.load(args.train_features),
        np.load(args.train_labels),
        np.load(args.eval_features),
        np.load(args.eval_labels),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CIoEBT baseline for ear recognition"
    )
    parser.add_argument("--dataset-root", default=str(config.data_root), help="Dataset root containing train/ class folders.")
    parser.add_argument("--train-folder", default=config.train_folder, help="Training folder name under dataset root.")    
    parser.add_argument("--out-dir", default=str(config.out_dir), help="Output directory.")
    parser.add_argument("--caffe-deploy", default=str(config.caffe_deploy_prototxt), help="Path to CaffeNet deploy.prototxt.")
    parser.add_argument("--caffe-weights", default=str(config.caffe_weights), help="Path to CaffeNet .caffemodel weights.")
    parser.add_argument("--caffe-proto", default=str(config.caffe_proto_local), help="Path to local caffe.proto.")
    parser.add_argument("--caffenet-checkpoint", default=None, help="Optional checkpoint to load before feature extraction.")
    parser.add_argument("--skip-caffenet-training", action="store_true", help="Skip supervised CaffeNet head fine-tuning and only extract features.")
    parser.add_argument("--unfreeze-backbone", action="store_true", help="Fine-tune the converted CaffeNet backbone instead of training only the new head.")

    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--num-workers", type=int, default=config.num_workers)
    parser.add_argument("--lr", type=float, default=config.lr)
    parser.add_argument("--weight-decay", type=float, default=config.weight_decay)
    parser.add_argument("--eval-ratio", type=float, default=config.eval_ratio)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--depml-u", type=float, default=config.depml_u)
    parser.add_argument("--depml-l", type=float, default=config.depml_l)
    parser.add_argument("--depml-cycles", type=int, default=config.depml_cycles_m)
    parser.add_argument("--depml-eta", type=float, default=config.depml_eta)
    parser.add_argument("--depml-gamma", type=float, default=config.depml_gamma)
    parser.add_argument("--knn-k", type=int, default=config.depml_knn_k)

    parser.add_argument("--comb-orders", nargs="+", default=[str(x) for x in config.comb_orders], help="Comb-filter orders, e.g. 6 8 10 12 or 6,8,10,12.")
    parser.add_argument("--comb-aug-factor", type=int, default=config.comb_aug_factor)
    parser.add_argument("--comb-passes", type=int, default=config.comb_multi_pass)
    parser.add_argument("--no-comb-protection", action="store_true", help="Fit DEP-ML directly on CaffeNet features.")
    parser.add_argument("--real-valued-earcode", action="store_true", help="Keep protected templates real-valued instead of binary.")

    parser.add_argument("--save-feature-arrays", action="store_true", help="Save extracted/processed feature arrays as .npy files.")
    parser.add_argument("--from-feature-arrays", action="store_true", help="Skip CaffeNet entirely and run DEP-ML from saved .npy arrays.")
    parser.add_argument("--train-features", default=None)
    parser.add_argument("--train-labels", default=None)
    parser.add_argument("--eval-features", default=None)
    parser.add_argument("--eval-labels", default=None)

    args = parser.parse_args()
    update_config_from_args(args)
    set_seed(config.seed)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    orders = parse_orders(args.comb_orders)

    if args.from_feature_arrays:
        X_train, y_train, X_eval, y_eval = load_feature_arrays(args)
        num_classes = int(len(np.unique(y_train)))
    else:
        train_ds, eval_ds, train_loader, eval_loader = build_loaders(args)
        num_classes = len(train_ds.class_to_idx)

        if args.skip_caffenet_training:
            model = CaffeNetCaffe2TorchEar(num_classes=num_classes).to(device)
        else:
            model, _history = train_caffenet_stage(train_loader, eval_loader, num_classes, device, out_dir)

        if args.caffenet_checkpoint:
            ckpt = torch.load(args.caffenet_checkpoint, map_location=device)
            state_dict = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded CaffeNet checkpoint: {args.caffenet_checkpoint}")

        X_train, y_train = extract_embeddings(model, train_loader, device)
        X_eval, y_eval = extract_embeddings(model, eval_loader, device)

        if args.save_feature_arrays:
            np.save(out_dir / "caffenet_train_features.npy", X_train)
            np.save(out_dir / "caffenet_train_labels.npy", y_train)
            np.save(out_dir / "caffenet_eval_features.npy", X_eval)
            np.save(out_dir / "caffenet_eval_labels.npy", y_eval)

    summaries = []
    if args.no_comb_protection:
        summary = fit_depml_stage(
            X_train,
            y_train,
            X_eval,
            y_eval,
            out_dir=out_dir / "depml_no_comb",
            apply_comb_protection=False,
            comb_order=0,
            save_arrays=args.save_feature_arrays,
        )
        summaries.append(summary)
    else:
        for order in orders:
            summary = fit_depml_stage(
                X_train,
                y_train,
                X_eval,
                y_eval,
                out_dir=out_dir / f"comb_order_{order}",
                apply_comb_protection=True,
                comb_order=order,
                save_arrays=args.save_feature_arrays,
            )
            summaries.append(summary)

    run_summary = {
        "baseline": "omara2025_cioebt_depml_caffenet",
        "num_classes": num_classes,
        "train_features": list(np.asarray(X_train).shape),
        "eval_features": list(np.asarray(X_eval).shape),
        "results": summaries,
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
