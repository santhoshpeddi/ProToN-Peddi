# Mehta et al. (2024): Ensemble-Based Hybrid Transfer Approach for 2D Ear Recognition

This directory contains our implementation of the baseline derived from:

> Ravishankar Mehta, Akbar Sheikh-Akbari, and Koushlendra Kumar Singh,  
> **Ensemble-based hybrid transfer approach for an effective 2D ear recognition system**,  
> *IEEE Access*, vol. 12, pp. 155733-155746, 2024.

## What is implemented here

The implementation in this folder provides:

- a clean training entry point via `main.py`;
- dataset loading and preprocessing in `dataset.py`;
- model definitions in `model.py`;
- training and ensemble evaluation utilities in `train.py`;
- experiment defaults in `config.py`.

## Implementation scope

In our comparative study, we preserve the VGG-based hybrid transfer baseline design and re-run it under the unified evaluation protocol used in our paper.

To keep the comparison fair and reproducible:

- the architectural family and overall training strategy are kept aligned with the source paper;
- explicitly available training settings are preserved where possible;
- when dataset splits or protocol details differ from our study, the method is re-executed under our unified setup and the adaptation is documented here.

## Dataset structure expected by this code

The current implementation expects the dataset root to contain a class-folder structure under `train/`:

```text
/path/to/dataset/
└── train/
    ├── class_001/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── class_002/
    │   └── ...
    └── ...
```

`main.py` creates an internal train/eval split from the images inside each class folder using `eval_ratio`.

## Files in this folder

```text
Ensemble[26]/
├── README.md
├── main.py
├── config.py
├── dataset.py
├── model.py
└── train.py
```

## How to run

Train and evaluate the full VGG16+VGG19 ensemble workflow:

```bash
python main.py   --dataset-root /path/to/dataset   --out-dir /path/to/outputs   --weights-dir /path/to/pretrained_vgg_weights   --mode ensemble
```

Train only the VGG16 branch:

```bash
python main.py   --dataset-root /path/to/dataset   --out-dir /path/to/outputs   --weights-dir /path/to/pretrained_vgg_weights   --mode vgg16
```

Train only the VGG19 branch:

```bash
python main.py   --dataset-root /path/to/dataset   --out-dir /path/to/outputs   --weights-dir /path/to/pretrained_vgg_weights   --mode vgg19
```

## Main command-line arguments

| Argument | Description |
|---|---|
| `--dataset-root` | Root directory of the dataset |
| `--out-dir` | Directory to save checkpoints, logs, and plots |
| `--weights-dir` | Directory containing optional local `vgg16.pth` and `vgg19.pth` files |
| `--train-folder` | Folder containing class-wise training images |
| `--mode` | Train `vgg16`, `vgg19`, or the full `ensemble` |
| `--epochs` | Number of training epochs |
| `--batch-size` | Batch size |
| `--lr` | Learning rate |
| `--dropout` | Dropout applied before the classifier head |
| `--weight-decay` | L2 regularization coefficient |
| `--eval-ratio` | Fraction of training images per class used for internal evaluation |
| `--image-size` | Input image size |
| `--seed` | Random seed |
| `--num-workers` | Number of dataloader workers |
| `--device` | Device to use, such as `cpu` or `cuda` |
| `--w1` | Ensemble weight for VGG16 |
| `--w2` | Ensemble weight for VGG19 |
| `--no-pretrained` | Disable pretrained initialization |
| `--allow-torchvision-fallback` | Allow torchvision to download fallback ImageNet weights when local weights are not found |

## Output files

The script creates:

- `checkpoints/vgg16_svm_best.pth` — best VGG16 checkpoint by evaluation accuracy;
- `checkpoints/vgg19_svm_best.pth` — best VGG19 checkpoint by evaluation accuracy;
- `metrics/vgg16_svm_history.csv` — epoch-wise VGG16 training log;
- `metrics/vgg19_svm_history.csv` — epoch-wise VGG19 training log;
- `metrics/vgg16_svm_summary.json` — compact VGG16 run summary;
- `metrics/vgg19_svm_summary.json` — compact VGG19 run summary;
- `metrics/ensemble_summary.json` — weighted-ensemble evaluation summary;
- `curves/vgg16_svm_loss_curve.png` and `curves/vgg16_svm_accuracy_curve.png`;
- `curves/vgg19_svm_loss_curve.png` and `curves/vgg19_svm_accuracy_curve.png`.

## Example output layout

```text
outputs/
├── checkpoints/
│   ├── vgg16_svm_best.pth
│   └── vgg19_svm_best.pth
├── metrics/
│   ├── vgg16_svm_history.csv
│   ├── vgg16_svm_summary.json
│   ├── vgg19_svm_history.csv
│   ├── vgg19_svm_summary.json
│   └── ensemble_summary.json
└── curves/
    ├── vgg16_svm_loss_curve.png
    ├── vgg16_svm_accuracy_curve.png
    ├── vgg19_svm_loss_curve.png
    └── vgg19_svm_accuracy_curve.png
```

## Reproducibility notes

- Set `--seed` for deterministic splitting and repeatable experiments.
- When using local pretrained weights, place `vgg16.pth` and `vgg19.pth` inside the path passed to `--weights-dir`.
- Some augmentation probabilities and certain operational details are implementation-level choices for reproducible execution when the original paper does not specify every release-ready detail.


## Citation

```bibtex
@article{mehta2024ensemble,
  title     = {Ensemble-based hybrid transfer approach for an effective 2D ear recognition system},
  author    = {Mehta, Ravishankar and Sheikh-Akbari, Akbar and Singh, Koushlendra Kumar},
  journal   = {IEEE Access},
  volume    = {12},
  pages     = {155733--155746},
  year      = {2024},
  publisher = {IEEE},
  doi       = {10.1109/ACCESS.2024.3485514}
}
```
