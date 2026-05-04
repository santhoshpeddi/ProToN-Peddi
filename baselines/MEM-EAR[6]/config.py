"""Configuration defaults for the UERC 2023 MEM-Ear participant baseline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class Config:
    seed: int = 42
    split_seed: int = 32

    # Dataset is preprocessed and aligned as described in the baseline paper, 
    # and stored in this form for faster computation.
    dataset_root: Path = Path("data/uerc23_aligned_dataset_MEM-EAR[6]")
    train_folder: str = "train"
    output_dir: Path = Path("outputs/memear_fusion")
    demographics_csv: Path | None = None
    allowed_ext: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
    validation_ratio: float = 0.20

    # Model configuration
    num_classes: int = 0
    convnext_input_size: int = 224
    efficientnet_input_size: int = 224
    iresnet_input_size: int = 112
    embedding_dim: int = 512
    fusion_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    normalize_embeddings: bool = True
    dropout: float = 0.5

    # Augmentation and normalization
    train_horizontal_flip: bool = True
    train_color_jitter: bool = True
    imagenet_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    imagenet_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # Optimization
    epochs: int = 150
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    val_interval: int = 5
    use_scheduler: bool = False
    scheduler_step: int = 10
    scheduler_gamma: float = 0.1
    device: str = "cuda"

    use_demographic_weights: bool = False

    # local pretrained weights.
    convnext_pretrained_path: str = "convnext_tiny.pth"
    efficientnet_pretrained_path: str = "efficientnet_b3.pth"
    iresnet_pretrained_path: str = "r100_ms1mv2.pth"

    @property
    def train_dir(self) -> Path:
        return self.dataset_root / self.train_folder

    @property
    def csv_dir(self) -> Path:
        return self.output_dir / "csv"

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def metrics_csv(self) -> Path:
        return self.output_dir / "metrics" / "history.csv"

    @property
    def summary_json(self) -> Path:
        return self.output_dir / "metrics" / "summary.json"

    @property
    def best_uerc_model(self) -> Path:
        return self.checkpoint_dir / "best_model.pth"
