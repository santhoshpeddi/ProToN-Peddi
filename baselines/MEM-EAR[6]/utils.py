from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import MEMEarDataset


def get_identities(root: str | Path) -> List[str]:
    root = Path(root)
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def collect_samples(root: str | Path, allowed_ext: Iterable[str]) -> Tuple[List[Dict], set]:
    root = Path(root)
    samples = []
    identities = get_identities(root)
    allowed = tuple(ext.lower() for ext in allowed_ext)

    for identity in identities:
        identity_dir = root / identity
        for file in sorted(identity_dir.iterdir()):
            if file.is_file() and file.name.lower().endswith(allowed):
                samples.append({"image_path": str(file), "identity": str(identity)})

    return samples, set(identities)


def load_demographics(demo_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(demo_csv, sep=None, engine="python")
    df.columns = [c.lower() for c in df.columns]
    required = {"cid", "gender", "ethnicity"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Demographics CSV must contain columns {sorted(required)}")
    return df.set_index("cid")


def attach_demographics(samples: List[Dict], demo_df: pd.DataFrame) -> List[Dict]:
    enriched = []
    for sample in samples:
        try:
            cid = int(sample["identity"])
        except ValueError as exc:
            raise RuntimeError("Demographic attachment expects numeric identity folders matching cid") from exc

        if cid not in demo_df.index:
            raise RuntimeError(f"Missing demographics for cid={cid}")

        row = demo_df.loc[cid]
        gender = str(row["gender"]).lower()
        ethnicity = str(row["ethnicity"]).lower()
        enriched.append({
            **sample,
            "cid": cid,
            "gender": gender,
            "ethnicity": ethnicity,
            "demographic_group": f"{ethnicity}_{gender}",
        })
    return enriched


def _ensure_demographics_column(samples: List[Dict]) -> None:
    for sample in samples:
        sample.setdefault("cid", sample["identity"])
        sample.setdefault("gender", "unknown")
        sample.setdefault("ethnicity", "unknown")
        sample.setdefault("demographic_group", "unknown")


def write_csv(samples: List[Dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "identity", "cid", "gender", "ethnicity", "demographic_group"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)


def generate_csvs(cfg, allow_identity_overlap: bool = False) -> Tuple[Path, Path]:
    cfg.csv_dir.mkdir(parents=True, exist_ok=True)
    train_samples, train_ids = collect_samples(cfg.train_dir, cfg.allowed_ext)

    if not train_samples:
        raise RuntimeError(f"No training images found under {cfg.train_dir}")

    if cfg.use_demographic_weights:
        if cfg.demographics_csv is None:
            raise RuntimeError("--use-demographic-weights requires --demographics-csv")
        demo_df = load_demographics(cfg.demographics_csv)
        train_samples = attach_demographics(train_samples, demo_df)
    else:
        _ensure_demographics_column(train_samples)

    train_csv = cfg.csv_dir / "train.csv"
    write_csv(train_samples, train_csv)
    return train_csv


def compute_demographic_weights(train_csv: str | Path) -> Dict[str, float]:
    df = pd.read_csv(train_csv)
    counts = df["demographic_group"].value_counts()
    proportions = counts / counts.sum()
    weights = 1.0 / proportions
    weights = weights / weights.mean()
    return weights.to_dict()


def split_train_val_imagewise(
    train_csv: str | Path,
    out_dir: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Path, Path, Dict[str, int]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    df = pd.read_csv(train_csv)
    if df.empty:
        raise RuntimeError(f"train_csv is empty: {train_csv}")
    if "identity" not in df.columns:
        raise RuntimeError("identity column missing in train_csv")

    df = df.dropna(subset=["identity"])
    df["identity"] = df["identity"].astype(str).str.strip()
    df = df[df["identity"] != ""]
    counts = df["identity"].value_counts()

    multi_img_ids = counts[counts >= 2].index
    single_img_ids = counts[counts == 1].index
    df_multi = df[df["identity"].isin(multi_img_ids)].copy()
    df_single = df[df["identity"].isin(single_img_ids)].copy()

    if len(df_multi) == 0:
        print("[WARN] No identities with >=2 images. Using all data for training and empty validation.")
        train_df = df.copy()
        val_df = df.iloc[0:0].copy()
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(splitter.split(df_multi, df_multi["identity"]))
        train_df = df_multi.iloc[train_idx].copy()
        val_df = df_multi.iloc[val_idx].copy()
        train_df = pd.concat([train_df, df_single], ignore_index=True)

    identities = sorted(train_df["identity"].unique())
    id2label = {identity: idx for idx, identity in enumerate(identities)}
    train_df["label"] = train_df["identity"].map(id2label)
    val_df = val_df[val_df["identity"].isin(id2label)].copy()
    val_df["label"] = val_df["identity"].map(id2label)

    train_out = out_dir / "train_split.csv"
    val_out = out_dir / "val_split.csv"
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)

    with (out_dir / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=2)

    print(f"[INFO] Train images: {len(train_df)} | Val images: {len(val_df)} | Classes: {len(id2label)}")
    return train_out, val_out, id2label


def build_dataloaders(cfg, train_transforms, eval_transforms, train_csv, val_csv):
    train_dataset = MEMEarDataset(train_csv, train_transforms, is_train=True)
    val_dataset = MEMEarDataset(val_csv, eval_transforms, is_train=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, 2 * cfg.batch_size),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return train_loader, val_loader


def build_train_transforms(cfg):
    common_aug = []
    if cfg.train_horizontal_flip:
        common_aug.append(transforms.RandomHorizontalFlip(p=0.5))
    if cfg.train_color_jitter:
        common_aug.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))

    return {
        "convnext": transforms.Compose([
            transforms.Resize((cfg.convnext_input_size, cfg.convnext_input_size)),
            *common_aug,
            transforms.ToTensor(),
            transforms.Normalize(cfg.imagenet_mean, cfg.imagenet_std),
        ]),
        "efficientnet": transforms.Compose([
            transforms.Resize((cfg.efficientnet_input_size, cfg.efficientnet_input_size)),
            *common_aug,
            transforms.ToTensor(),
            transforms.Normalize(cfg.imagenet_mean, cfg.imagenet_std),
        ]),
        "iresnet": transforms.Compose([
            transforms.Resize((cfg.iresnet_input_size, cfg.iresnet_input_size)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.imagenet_mean, cfg.imagenet_std),
        ]),
    }


def build_eval_transforms(cfg):
    return {
        "convnext": transforms.Compose([
            transforms.Resize((cfg.convnext_input_size, cfg.convnext_input_size)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.imagenet_mean, cfg.imagenet_std),
        ]),
        "efficientnet": transforms.Compose([
            transforms.Resize((cfg.efficientnet_input_size, cfg.efficientnet_input_size)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.imagenet_mean, cfg.imagenet_std),
        ]),
        "iresnet": transforms.Compose([
            transforms.Resize((cfg.iresnet_input_size, cfg.iresnet_input_size)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.imagenet_mean, cfg.imagenet_std),
        ]),
    }
