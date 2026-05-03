# Aiadi et al. (2023): MDFNet for Unsupervised Ear-Print Recognition

This directory contains our implementation of the baseline derived from:

> Oussama Aiadi, Belal Khaldi, and Cheraa Saadeddine,  
> **MDFNet: an unsupervised lightweight network for ear print recognition**,  
> *Journal of Ambient Intelligence and Humanized Computing*, vol. 14, no. 10, pp. 13773-13786, 2023.

## What is implemented here

The implementation in this folder provides:

- a clean training entry point via `main.py`;
- dataset loading in `dataset.py`;
- the MDFNet unsupervised feature extractor in `mdfnet.py`;
- training utilities in `train.py`;
- experiment defaults and MDFNet subset settings in `config.py`.

## Implementation scope

In our comparative study, we preserve the MDFNet feature-extraction design and re-run it under the unified evaluation protocol used in our paper.

To keep the comparison fair and reproducible:

- the architectural family and overall training strategy are kept aligned with the source paper;
- explicitly available training settings are preserved where possible;
- when dataset splits or protocol details differ from our study, the method is re-executed under our unified setup and the adaptation is documented here.

## MDFNet subset configurations

The Paper states the six subset settings used by the MDFNet pipeline:

| Subset | Number of PCA filters `M` | Filter size `p` | Block size |
|---|---:|---:|---:|
| 1 | 5 | 5 | 16 |
| 2 | 5 | 7 | 32 |
| 3 | 9 | 9 | 16 |
| 4 | 9 | 9 | 100 |
| 5 | 9 | 9 | 25 |
| 6 | 7 | 7 | 100 |

Subsets can be changed using `--subset-id`.

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
MDFNet[29]/
├── README.md
├── main.py
├── config.py
├── dataset.py
├── mdfnet.py
└── train.py
```

## How to run

From this directory:

```bash
python main.py \
  --dataset-root /path/to/dataset \
  --out-dir /path/to/outputs \
  --subset-id 3
```

Change SVM and loading settings:

```bash
python main.py \
  --dataset-root /path/to/dataset \
  --out-dir /path/to/outputs \
  --subset-id 5 \
  --batch-size 64 \
  --svm-c 1.0 \
  --svm-max-iter 10000
```

## Main command-line arguments

| Argument | Description |
|---|---|
| `--dataset-root` | Root directory of the dataset |
| `--out-dir` | Directory to save checkpoints, metrics, and optional features |
| `--train-folder` | Folder containing class-wise training images |
| `--use-internal-eval` | Use an internal class-wise eval split |
| `--eval-ratio` | Fraction of images per class used for internal evaluation |
| `--subset-id` | MDFNet subset ID from `1` to `6` |
| `--batch-size` | Batch size used while collecting images for feature extraction |
| `--num-workers` | Number of dataloader workers |
| `--seed` | Random seed for deterministic splitting |
| `--image-size` | Square grayscale input size; MDFNet uses `96` by default |
| `--device` | Device used for convolution operations: `auto`, `cpu`, or `cuda` |
| `--svm-c` | Linear SVM regularization parameter |
| `--svm-max-iter` | Maximum number of LinearSVC iterations |
| `--save-features` | Save extracted train/eval MDFNet descriptors as `.npy` files |

## Output files

The script creates:

- `checkpoints/mdfnet_svm.pkl` — trained MDFNet + SVM model;
- `checkpoints/mdfnet_filters.npz` — learned PCA filter bank;
- `metrics/history.csv` — compact train/eval accuracy log;
- `metrics/summary.json` — run configuration and final metrics;
- `metrics/classification_report.json` — class-wise evaluation report;
- `features/*.npy` — optional extracted features and labels when `--save-features` is used.

## Example output layout

```text
outputs/
├── checkpoints/
│   ├── mdfnet_svm.pkl
│   └── mdfnet_filters.npz
├── metrics/
│   ├── history.csv
│   ├── summary.json
│   └── classification_report.json
└── features/
    ├── train_features.npy
    ├── train_labels.npy
    ├── eval_features.npy
    └── eval_labels.npy
```

## Reproducibility notes

- Set `--seed` for deterministic splitting.
- The SVM is supervised and is trained only after MDFNet descriptors are extracted.

## Citation

```bibtex
@article{aiadi2023mdfnet,
  title     = {MDFNet: an unsupervised lightweight network for ear print recognition},
  author    = {Aiadi, Oussama and Khaldi, Belal and Saadeddine, Cheraa},
  journal   = {Journal of Ambient Intelligence and Humanized Computing},
  volume    = {14},
  number    = {10},
  pages     = {13773--13786},
  year      = {2023},
  publisher = {Springer},
  doi       = {10.1007/s12652-022-04028-z}
}
```
