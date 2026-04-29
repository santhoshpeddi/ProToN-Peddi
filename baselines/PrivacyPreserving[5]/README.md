# Chowdhury et al. (2022): Privacy Preserving Ear Recognition with Transfer Learning

This directory contains our implementation of the baseline derived from:

> Debbrota Paul Chowdhury, Sambit Bakshi, Chiara Pero, Gustavo Olague, and Pankaj Kumar Sa,  
> **Privacy Preserving Ear Recognition System Using Transfer Learning in Industry 4.0**,  
> *IEEE Transactions on Industrial Informatics*, vol. 19, no. 5, pp. 6408-6417, 2022.

## What is implemented here

The implementation in this folder provides:

- a clean training entry point via `main.py`;
- dataset loading and augmentation in `dataset.py`;
- model definitions in `model.py`;
- training and logging utilities in `train.py`;
- experiment defaults in `config.py`.

## Implementation scope

In our comparative study, we preserve the DenseNet-based transfer-learning baseline design and re-run it under the unified evaluation protocol used in our paper.

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
PrivacyPreserving[5]/
├── README.md
├── main.py
├── config.py
├── dataset.py
├── model.py
└── train.py
```

## How to run

From this directory:

```bash
python main.py \
  --dataset-root /path/to/dataset \
  --out-dir /path/to/outputs \
  --weights-path /path/to/densenet161.pth
```

Run without local ImageNet weights:

```bash
python main.py \
  --dataset-root /path/to/dataset \
  --out-dir /path/to/outputs \
  --no-local-weights
```

## Main command-line arguments

| Argument | Description |
|---|---|
| `--dataset-root` | Root directory of the dataset |
| `--out-dir` | Directory to save checkpoints, logs, and plots |
| `--weights-path` | Path to local DenseNet-161 ImageNet weights |
| `--train-folder` | Folder containing class-wise training images |
| `--epochs` | Number of training epochs |
| `--batch-size` | Batch size |
| `--lr` | Initial learning rate |
| `--momentum` | SGD momentum |
| `--weight-decay` | L2 regularization coefficient |
| `--eval-ratio` | Fraction of training images per class used for internal evaluation |
| `--image-size` | Input image size |
| `--seed` | Random seed |
| `--num-workers` | Number of dataloader workers |
| `--device` | Device to use, such as `cpu` or `cuda` |
| `--use-class-weights` | Enable class-weighted cross-entropy |
| `--no-local-weights` | Train without local ImageNet initialization |

## Output files

The script produces:

- `checkpoints/best_model.pt` — best checkpoint saved during training;
- `checkpoints/last_model.pt` — final checkpoint at the end of training;
- `metrics/history.csv` — epoch-wise training log;
- `metrics/summary.json` — compact summary of the run;
- `curves/loss_curve.png` — training/evaluation loss curve;
- `curves/accuracy_curve.png` — training/evaluation accuracy curve.

## Example output layout

```text
outputs/
├── checkpoints/
│   ├── best_model.pt
│   └── last_model.pt
├── metrics/
│   ├── history.csv
│   └── summary.json
└── curves/
    ├── loss_curve.png
    └── accuracy_curve.png
```

## Reproducibility notes

- Set `--seed` for deterministic splitting and repeatable experiments.
- When using local pretrained weights, pass the DenseNet-161 ImageNet checkpoint through `--weights-path`.
- Some augmentation probabilities and certain operational details are implementation-level choices for reproducible execution when the original paper does not specify every release-ready detail.

## Citation

```bibtex
@article{chowdhury2022privacy,
  title     = {Privacy preserving ear recognition system using transfer learning in industry 4.0},
  author    = {Chowdhury, Debbrota Paul and Bakshi, Sambit and Pero, Chiara and Olague, Gustavo and Sa, Pankaj Kumar},
  journal   = {IEEE Transactions on Industrial Informatics},
  volume    = {19},
  number    = {5},
  pages     = {6408--6417},
  year      = {2022},
  publisher = {IEEE},
  doi={10.1109/TII.2022.3196343}
}
```
