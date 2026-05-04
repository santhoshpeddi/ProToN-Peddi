from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration for the CIoEBT baseline."""

    # Dataset
    data_root: Path = Path("data/uerc23_dataset")
    train_folder: str = "train"
    eval_ratio: float = 0.40
    seed: int = 42

    # Output
    out_dir: Path = Path("outputs")
    history_csv: str = "training_history.csv"
    best_ckpt: str = "caffenet_best.pth"
    last_ckpt: str = "caffenet_last.pth"

    # CaffeNet / Caffe conversion files
    caffe_deploy_prototxt: Path = Path("caffe/models/bvlc_reference_caffenet/deploy.prototxt")
    caffe_proto_local: Path = Path("caffe/src/caffe/proto/caffe.proto")
    # Download the pretrained weights from the BVLC/Caffe repository
    caffe_weights: Path = Path("caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel")

    # Stage 1: CaffeNet classifier fine-tuning
    train_caffenet_stage1: bool = True
    freeze_backbone: bool = True
    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Caffe preprocessing
    caffe_resize: int = 256
    caffe_crop: int = 227
    caffe_bgr_mean: tuple[float, float, float] = (104.0, 117.0, 123.0)

    # Eq. 12 grayscale normalization
    use_eq12_gray_norm: bool = True
    nmin: float = 0.0
    nmax: float = 255.0

    # DEP-ML (Eq(3)-(7)) stage
    depml_u: float = 1.0
    depml_l: float = 4.0
    depml_cycles_m: int = 15
    depml_eta: float = 1e-3
    depml_gamma: float = 1.0
    depml_use_eq7: bool = True
    depml_knn_k: int = 3

    # Comb-filter (Eq(8)-(11)) stage
    comb_orders: tuple[int, ...] = (6, 8, 10, 12)
    comb_aug_factor: int = 10
    comb_multi_pass: int = 3
    earcode_binary: bool = True


config = Config()
