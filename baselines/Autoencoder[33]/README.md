# Pal et al. (2025): Shape-Focused Autoencoder for Ear Biometrics

This directory contains our implementation of the baseline derived from:

> Hrithik Pal, Soubhik Acharya, Priti Paul, Bitan Misra, Jungpil Shin, and Nilanjan Dey,  
> **Exploration of a shape-focused autoencoder for improved ear biometrics**,  
> *Multimedia Tools and Applications*, vol. 84, no. 35, pp. 44897-44919, 2025.

## What is implemented here

The implementation in this folder provides:

- a clean training entry point via `main.py`;
- shape-based feature extraction in `features.py`;
- dataset preparation and feature standardization in `dataset.py`;
- model definitions in `model.py`;
- training and logging utilities in `train.py`;
- experiment defaults in `config.py`.

## Implementation scope

In our comparative study, we preserve the shape-feature-driven baseline design and re-run it under the unified evaluation protocol used in our paper.

To keep the comparison fair and reproducible:

- the architectural family and overall training strategy are kept aligned with the source paper;
- explicitly available training settings are preserved where possible;
- when dataset splits or protocol details differ from our study, the method is re-executed under our unified setup and the adaptation is documented here.

## Dataset structure expected by this code

The current implementation expects a class-folder structure under `train/`:

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
Autoencoder[33]/
├── README.md
├── main.py
├── config.py
├── dataset.py
├── features.py
├── model.py
└── train.py
```

## How to run

From this directory:

```bash
python main.py --dataset-root /path/to/dataset --out-dir /path/to/outputs
```

Change common training settings:

```bash
python main.py   --dataset-root /path/to/dataset   --out-dir /path/to/outputs   --epochs 50   --batch-size 32   --lr 0.001
```

## Main command-line arguments

| Argument | Description |
|---|---|
| `--dataset-root` | Root directory of the dataset |
| `--out-dir` | Directory to save model weights, logs, plots, and extracted features |
| `--train-folder` | Folder containing class-wise training images |
| `--epochs` | Number of training epochs |
| `--batch-size` | Batch size |
| `--lr` | Adam learning rate |
| `--eval-ratio` | Fraction of training images per class used for internal evaluation |
| `--seed` | Random seed |
| `--num-workers` | Number of dataloader workers |
| `--device` | Device to use, such as `cpu` or `cuda` |
| `--disable-augmentations` | Turn off noise / contrast / gamma feature augmentation |
| `--disable-standard-scaler` | Skip the StandardScaler preprocessing step |

## Output files

The script creates:

- `model_autoencoder_shape.pt` — final trained model checkpoint;
- `model_best_autoencoder_shape.pt` — best checkpoint by evaluation accuracy;
- `standard_scaler.joblib` — fitted feature scaler from the training split;
- `label_encoder.joblib` — label encoder for class names;
- `train_classes.json` — serialized class list used during training;
- `train_features.csv` — extracted and scaled training features;
- `eval_features.csv` — extracted and scaled evaluation features;
- `training_history.csv` — epoch-wise training log;
- `summary.json` — compact run summary;
- `loss_curve.png` and `accuracy_curve.png` — training curves.

## Example output layout

```text
outputs/
├── model_autoencoder_shape.pt
├── model_best_autoencoder_shape.pt
├── standard_scaler.joblib
├── label_encoder.joblib
├── train_classes.json
├── train_features.csv
├── eval_features.csv
├── training_history.csv
├── summary.json
├── loss_curve.png
└── accuracy_curve.png
```

## Reproducibility notes

- Set `--seed` for deterministic splitting and repeatable experiments.
- Some augmentation probabilities and certain operational details are implementation-level choices for reproducible execution when the original paper does not specify every release-ready detail.

## Citation

```bibtex
@article{pal2025exploration,
  title     = {Exploration of a shape-focused autoencoder for improved ear biometrics},
  author    = {Pal, Hrithik and Acharya, Soubhik and Paul, Priti and Misra, Bitan and Shin, Jungpil and Dey, Nilanjan},
  journal   = {Multimedia Tools and Applications},
  volume    = {84},
  number    = {35},
  pages     = {44897--44919},
  year      = {2025},
  publisher = {Springer},
  doi       = {10.1007/s11042-025-20905-z}
}
```
