# Alshazly et al. (2021): Explainable Ear Recognition with Deep Residual Networks

This directory contains our implementation of the baseline derived from:

> Hammam Alshazly, Christoph Linse, Erhardt Barth, Sahar Ahmed Idris, and Thomas Martinetz,  
> **Towards Explainable Ear Recognition Systems Using Deep Residual Networks**,  
> *IEEE Access*, vol. 9, pp. 122254-122273, 2021.

## What is implemented here

The implementation in this folder provides:

- a clean training entry point via `main.py`;
- dataset loading and augmentation in `dataset.py`;
- model definitions in `model.py`;
- experiment defaults in `config.py`.

## Implementation scope

In our comparative study, we preserve the ResNet-based baseline design and re-run it under the unified evaluation protocol used in our paper.

To keep the comparison fair and reproducible:

- the architectural family and training strategy are kept aligned with the source paper;
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
alshazly2021_explainable_resnets/
├── README.md
├── main.py
├── config.py
├── dataset.py
└── model.py
```

## How to run

From this directory:

```bash
python main.py --dataset-root /path/to/dataset --variant resnet50
```

Train all supported backbone variants sequentially:

```bash
python main.py --dataset-root /path/to/dataset --train-all
```

Use local ImageNet pretrained weights:

```bash
python main.py \
  --dataset-root /path/to/dataset \
  --variant resnet101 \
  --use-local-pretrained \
  --weights-dir /path/to/resnet_weights
```

Change output location:

```bash
python main.py \
  --dataset-root /path/to/dataset \
  --variant resnet34 \
  --out-dir /path/to/outputs
```

## Main command-line arguments

| Argument | Description |
|---|---|
| `--dataset-root` | Root directory of the dataset |
| `--train-folder` | Folder containing class-wise training images |
| `--variant` | One of `resnet34`, `resnet50`, `resnet101`, `resnet152` |
| `--train-all` | Train all supported variants sequentially |
| `--epochs` | Number of training epochs |
| `--batch-size` | Batch size |
| `--lr` | Initial learning rate |
| `--weights-dir` | Directory containing optional local pretrained weights |
| `--use-local-pretrained` | Load local pretrained ResNet weights |
| `--eval-ratio` | Fraction of images per class used for internal evaluation |
| `--out-dir` | Directory to save checkpoints and metrics |
| `--seed` | Random seed |

## Output files

For each trained variant, the script creates:

- `checkpoints/best_<variant>.pth` — best checkpoint by validation accuracy;
- `metrics/<variant>_history.csv` — epoch-wise training log;
- `metrics/<variant>_summary.json` — compact summary of the run.

## Example output layout

```text
outputs/
├── checkpoints/
│   └── best_resnet50.pth
└── metrics/
    ├── resnet50_history.csv
    └── resnet50_summary.json
```

## Reproducibility notes

- Set `--seed` for deterministic splitting and repeatable experiments.
- When using local pretrained weights, place files such as `resnet50.pth` inside the path passed to `--weights-dir`.
- The script trains the full network end-to-end rather than freezing the backbone.

## Citation

```bibtex
@article{alshazly2021towards,
  title     = {Towards explainable ear recognition systems using deep residual networks},
  author    = {Alshazly, Hammam and Linse, Christoph and Barth, Erhardt and Idris, Sahar Ahmed and Martinetz, Thomas},
  journal   = {IEEE Access},
  volume    = {9},
  pages     = {122254--122273},
  year      = {2021},
  publisher = {IEEE},
  doi       = {10.1109/ACCESS.2021.3109441}
}
```
