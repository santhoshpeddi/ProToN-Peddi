from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ShapeAutoencoderConfig:
    """Configuration for the Pal et al. baseline."""

    dataset_root: Path = Path('data/uerc23_dataset')
    train_folder: str = 'train'
    out_dir: Path = Path('outputs')

    # Feature extraction
    gauss_blur_kernel: tuple[int, int] = (5, 5)
    canny_min_thresh: int = 50
    canny_max_thresh: int = 150
    num_feature_points: int = 100
    gftt_quality_level: float = 0.01
    gftt_min_distance: float = 5.0

    # Data usage & augmentation
    use_augmentations: bool = True
    aug_random_noise: bool = True
    aug_contrast: bool = True
    aug_gamma: bool = True
    aug_rotation: bool = False

    # Split & scaling
    eval_ratio: float = 0.30
    seed: int = 42
    use_standard_scaler: bool = True

    # Model hyperparameters
    enc_dims: tuple[int, int] = (128, 64)
    dec_dims: tuple[int, ...] = (64,)
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    num_workers: int = 0

    @property
    def model_weights_path(self) -> Path:
        return self.out_dir / 'model_autoencoder_shape.pt'

    @property
    def best_model_path(self) -> Path:
        return self.out_dir / 'model_best_autoencoder_shape.pt'

    @property
    def scaler_path(self) -> Path:
        return self.out_dir / 'standard_scaler.joblib'

    @property
    def label_encoder_path(self) -> Path:
        return self.out_dir / 'label_encoder.joblib'

    @property
    def train_classes_path(self) -> Path:
        return self.out_dir / 'train_classes.json'

    @property
    def train_features_csv(self) -> Path:
        return self.out_dir / 'train_features.csv'

    @property
    def eval_features_csv(self) -> Path:
        return self.out_dir / 'eval_features.csv'

    @property
    def history_csv(self) -> Path:
        return self.out_dir / 'training_history.csv'

    @property
    def summary_json(self) -> Path:
        return self.out_dir / 'summary.json'

    @property
    def loss_curve_png(self) -> Path:
        return self.out_dir / 'loss_curve.png'

    @property
    def acc_curve_png(self) -> Path:
        return self.out_dir / 'accuracy_curve.png'


DEFAULT_CONFIG = ShapeAutoencoderConfig()
