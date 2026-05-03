from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    """Configuration for the Mehta et al. ensemble baseline."""

    dataset_root: Path = Path("data/uerc23_dataset")
    out_dir: Path = Path("outputs")
    pretrained_vgg_dir: Path = Path("weights/vgg_imagenet")

    train_folder: str = "train"

    image_size: int = 150
    eval_ratio: float = 0.20
    seed: int = 42
    num_workers: int = 2

    batch_size: int = 16
    lr: float = 0.001
    epochs: int = 100
    dropout: float = 0.5
    weight_decay: float = 0.01

    ensemble_weight_vgg16: float = 0.5
    ensemble_weight_vgg19: float = 0.5

    use_pretrained: bool = True
    allow_torchvision_fallback: bool = False

    @property
    def vgg16_weights_path(self) -> Path:
        return self.pretrained_vgg_dir / "vgg16.pth"

    @property
    def vgg19_weights_path(self) -> Path:
        return self.pretrained_vgg_dir / "vgg19.pth"

    @property
    def checkpoints_dir(self) -> Path:
        return self.out_dir / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return self.out_dir / "metrics"

    @property
    def curves_dir(self) -> Path:
        return self.out_dir / "curves"


DEFAULT_CONFIG = TrainConfig()
