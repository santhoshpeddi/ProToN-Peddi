from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class TrainConfig:
    """Configuration for the Alshazly et al. (2021) ResNet baseline."""

    dataset_root: Path = Path("data/uerc23_dataset")
    train_folder: str = "train"

    variants: List[str] = field(default_factory=lambda: ["resnet34", "resnet50", "resnet101", "resnet152"])
    default_variant: str = "resnet50"
    weights_dir: Path = Path("weights/resnet_imagenet")
    use_local_pretrained: bool = False

    input_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    eval_ratio: float = 0.20
    seed: int = 42

    epochs: int = 150
    lr: float = 0.02
    step_size: int = 30
    gamma: float = 0.5
    momentum: float = 0.9
    weight_decay: float = 0.0

    out_dir: Path = Path("outputs")

    @property
    def ckpt_dir(self) -> Path:
        return self.out_dir / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return self.out_dir / "metrics"


DEFAULT_CONFIG = TrainConfig()
