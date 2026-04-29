from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    # Dataset is preprocessed and aligned as described in the baseline paper, 
    # and stored in this form for faster computation.
    dataset_root: Path = Path("data/uerc23_aligned_dataset_PrivacyPreserving[5]")
    out_dir: Path = Path("outputs")

    train_folder: str = "train"
    test_folder: str = "test"
    eval_on_test_folder: bool = True

    imagenet_weights_path: Path = Path("weights/densenet161.pth")
    use_local_imagenet_weights: bool = True
    use_class_weights: bool = False

    epochs: int = 70
    batch_size: int = 8
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-4

    min_lr: float = 1e-15
    lr_reduce_factor: float = 5.0
    lr_plateau_patience: int = 5

    image_size: int = 224
    eval_ratio: float = 0.2
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True

    p_hcrop: float = 0.5
    p_vcrop: float = 0.5
    p_hflip: float = 0.5
    p_blur: float = 0.3
    p_histeq: float = 0.2
    p_adapthist: float = 0.2
    p_rot30: float = 0.3
    p_shear: float = 0.3
    p_bright20: float = 0.3
    p_noise: float = 0.3

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
