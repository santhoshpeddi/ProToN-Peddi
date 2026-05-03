from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    dataset_root: Path = Path("data/uerc23_dataset")
    train_folder: str = "train"
    out_dir: Path = Path("outputs")

    hg_ckpt: Path | None = None
    use_aligner: bool = True

    epochs: int = 2
    batch_size: int = 2
    eval_batch_size: int = 6
    num_workers: int = 4
    pin_memory: bool = True
    eval_ratio: float = 0.2
    seed: int = 42

    image_size: int = 518
    embed_dim: int = 512
    backbone_name: str = "vit_large_patch14_dinov2"
    pretrained_backbone: bool = True
    backbone_checkpoint: Path | None = None

    lr: float = 1e-4
    loss_name: str = "elastic_cosface_plus"
    cosface_scale: float = 64.0
    cosface_margin: float = 0.35
    elastic_std: float = 0.05

    @property
    def ckpt_dir(self) -> Path:
        return self.out_dir / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return self.out_dir / "metrics"

    @property
    def curves_dir(self) -> Path:
        return self.out_dir / "curves"


DEFAULT_CONFIG = TrainConfig()
