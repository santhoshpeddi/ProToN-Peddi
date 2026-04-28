# El-Naggar and Bourlai (2022): Deep Learning Ear Recognition in Thermal Images

This directory contains our implementation of the baseline derived from:

> Susan El-Naggar and Thirimachos Bourlai,  
> **Exploring Deep Learning Ear Recognition in Thermal Images**,  
> *IEEE Transactions on Biometrics, Behavior, and Identity Science*, vol. 5, no. 1, pp. 64-75.

## What is implemented here

The implementation in this folder provides:

- a clean training entry point via `main.py`;
- dataset loading and augmentation in `dataset.py`;
- image transformations in `transforms.py`;
- model definitions in `model.py`;
- experiment defaults in `config.py`.

## Implementation scope

In our comparative study, we preserve the DenseNet-based baseline design and re-run it under the unified evaluation protocol used in our paper.

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
DenseNet[4]/
├── README.md
├── main.py
├── config.py
├── dataset.py
├── transforms.py
└── model.py
```

## How to run

From this directory:

```bash
python main.py --dataset-root /path/to/dataset
```

For a transfer-learning run with local DenseNet-161 ImageNet weights:

```bash
python main.py   --dataset-root /path/to/dataset   --use-local-pretrained   --weights-path /path/to/densenet161.pth
```

Disable early-layer freezing if needed:

```bash
python main.py   --dataset-root /path/to/dataset   --freeze-early-layers false
```

## Main command-line arguments

| Argument | Description |
|---|---|
| `--dataset-root` | Root directory of the dataset |
| `--train-folder` | Folder containing class-wise training images |
| `--epochs` | Number of training epochs |
| `--batch-size` | Batch size |
| `--lr` | Learning rate |
| `--weights-path` | Path to local DenseNet-161 ImageNet weights |
| `--use-local-pretrained` | Load local pretrained DenseNet weights |
| `--freeze-early-layers` | Freeze the earlier feature layers |
| `--unfreeze-last-denseblock` | Unfreeze the last dense block |
| `--eval-ratio` | Fraction of images per class used for internal evaluation |
| `--out-dir` | Directory to save outputs |
| `--seed` | Random seed |

## Output files

The script produces:

- `checkpoints/best_densenet161.pth` — best checkpoint by validation accuracy;
- `checkpoints/last_densenet161.pth` — final checkpoint;
- `metrics/training_history.csv` — epoch-wise training log;
- `metrics/run_summary.json` — compact run summary;
- `curves/loss_curve.png` and `curves/acc_curve.png` — training curves.

## Reproducibility notes

- Set `--seed` for deterministic splitting and repeatable experiments.
- For the closest match to the source paper, provide local DenseNet-161 ImageNet weights with `--use-local-pretrained`.


## Citation

```bibtex
@article{el2022exploring,
  title     = {Exploring deep learning ear recognition in thermal images},
  author    = {El-Naggar, Susan and Bourlai, Thirimachos},
  journal   = {IEEE Transactions on Biometrics, Behavior, and Identity Science},
  volume    = {5},
  number    = {1},
  pages     = {64--75},
  year      = {2022},
  publisher = {IEEE},
  doi       = {10.1109/TBIOM.2022.3218151}
}
```
