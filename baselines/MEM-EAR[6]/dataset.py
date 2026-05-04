from __future__ import annotations

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class MEMEarDataset(Dataset):
    def __init__(self, csv_file: str, transforms: dict, is_train: bool = True):
        self.df = pd.read_csv(csv_file)
        self.transforms = transforms
        self.is_train = is_train

        required_cols = {"image_path", "demographic_group", "identity"}
        if self.is_train:
            required_cols.add("label")

        missing = required_cols - set(self.df.columns)
        if missing:
            raise RuntimeError(f"CSV is missing required columns: {sorted(missing)}")

        unique_ids = sorted(self.df["identity"].astype(str).unique())
        self.identity_to_idx = {identity: idx for idx, identity in enumerate(unique_ids)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")

        images = {
            "convnext": self.transforms["convnext"](img),
            "efficientnet": self.transforms["efficientnet"](img),
            "iresnet": self.transforms["iresnet"](img),
        }

        if self.is_train:
            label = int(row["label"])
        else:
            label = self.identity_to_idx[str(row["identity"])]

        demographic_group = str(row.get("demographic_group", "unknown"))
        return images, torch.tensor(label, dtype=torch.long), demographic_group
