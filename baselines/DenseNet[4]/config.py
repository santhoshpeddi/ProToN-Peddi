from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    dataset_root: Path = Path('data/uerc23_dataset')
    train_folder: str = 'train'

    model_name: str = 'densenet161'
    weights_path: Path = Path('weights/densenet161.pth')
    use_local_pretrained: bool = False

    freeze_early_layers: bool = True
    unfreeze_last_denseblock: bool = True

    input_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    eval_ratio: float = 0.20
    seed: int = 42

    epochs: int = 20
    lr: float = 3e-4
    momentum: float = 0.9
    weight_decay: float = 0.0

    out_dir: Path = Path('outputs')

    @property
    def ckpt_dir(self) -> Path:
        return self.out_dir / 'checkpoints'

    @property
    def metrics_dir(self) -> Path:
        return self.out_dir / 'metrics'

    @property
    def curves_dir(self) -> Path:
        return self.out_dir / 'curves'


DEFAULT_CONFIG = TrainConfig()
