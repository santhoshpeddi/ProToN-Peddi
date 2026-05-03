# Emeršič et al. (2023): UERC 2023 Participant (ViTEar)

This directory contains our implementation of a participant (ViTEar) baseline associated with:

> Žiga Emeršič, Tetsushi Ohki, Muku Akasaka, Takahiko Arakawa, Soshi Maeda, Masora Okano, Yuya Sato, Anjith George, Sébastien Marcel, Iyyakutti Iyappan Ganapathi, and others,  
> **The Unconstrained Ear Recognition Challenge 2023: Maximizing Performance and Minimizing Bias**,  
> *2023 IEEE International Joint Conference on Biometrics (IJCB)*, pp. 1-10, 2023.

## What is implemented here

The implementation in this folder provides:

- a clean training entry point via `main.py`;
- dataset loading and augmentation in `dataset.py`;
- model definitions in `model.py`;
- Two-stack hourglass alignment module in `aligner.py`;
- margin-based metric-learning losses in `losses.py`;
- training utilities in `train.py`;
- experiment defaults in `config.py`.

## Implementation scope

In our comparative study, we preserve the ViTEar design and re-run it under the unified evaluation protocol used in our paper.

To keep the comparison fair and reproducible:

- the architectural family and overall training strategy are kept aligned with the source paper;
- the baseline preserves the main components including a ViT-Large DINOv2-style embedding model, hourglass-based ear alignment, and CosFace-family losses;
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
ViTEar[6]/
├── README.md
├── main.py
├── config.py
├── dataset.py
├── model.py
├── aligner.py
├── losses.py
├── train.py
└── ensemble_eval.py
```

## How to run

From this directory:

```bash
python main.py   --dataset-root /path/to/dataset   --out-dir /path/to/outputs
```

Enable the hourglass ear aligner:

```bash
python main.py   --dataset-root /path/to/dataset   --out-dir /path/to/outputs   --use-aligner   --hg-ckpt /path/to/hourglass_checkpoint.pt
```

## Main command-line arguments

| Argument | Description |
|---|---|
| `--dataset-root` | Root directory of the dataset |
| `--train-folder` | Folder containing class-wise training images |
| `--out-dir` | Directory to save checkpoints and metrics |
| `--epochs` | Number of training epochs |
| `--batch-size` | Training batch size |
| `--eval-batch-size` | Evaluation batch size |
| `--lr` | Learning rate |
| `--embed-dim` | Output embedding dimensionality |
| `--loss-name` | One of `cosface`, `elastic_cosface`, `elastic_cosface_plus` |
| `--backbone-name` | Timm backbone name |
| `--pretrained-backbone` / `--no-pretrained-backbone` | Enable or disable pretrained backbone initialization |
| `--backbone-checkpoint` | Optional local checkpoint for the backbone |
| `--use-aligner` | Enable frozen two-stack hourglass alignment |
| `--hg-ckpt` | Path to the hourglass checkpoint |
| `--eval-ratio` | Fraction of images per class used for internal evaluation |
| `--num-workers` | Number of dataloader workers |
| `--seed` | Random seed |
| `--device` | Device to use, such as `cpu` or `cuda` |

## Output files

For each run, the script creates:

- `checkpoints/best_<loss>.pth` - best checkpoint by evaluation accuracy;
- `checkpoints/last_<loss>.pth` - final checkpoint;
- `metrics/history.csv` - epoch-wise training log;
- `metrics/summary.json` - compact summary of the run;
- `curves/loss_curve.png` - loss curve;
- `curves/accuracy_curve.png` - accuracy curve.

## Example output layout

```text
outputs/
├── checkpoints/
│   ├── best_elastic_cosface_plus.pth
│   └── last_elastic_cosface_plus.pth
├── curves/
│   ├── accuracy_curve.png
│   └── loss_curve.png
└── metrics/
    ├── history.csv
    └── summary.json
```

## Reproducibility notes

- Set `--seed` for deterministic splitting and repeatable experiments.
- If pretrained backbone weights are unavailable or downloads are restricted, use `--no-pretrained-backbone` or provide `--backbone-checkpoint`.
- Some augmentation probabilities and certain operational details are implementation-level choices for reproducible execution when the original paper does not specify every release-ready detail.

## Citation

```bibtex
@inproceedings{emervsivc2023unconstrained,
  title={The unconstrained ear recognition challenge 2023: Maximizing performance and minimizing bias},
  author={Emersic, Z and Ohki, Tetsushi and Akasaka, Muku and Arakawa, Takahiko and Maeda, Soshi and Okano, Masora and Sato, Yuya and George, Anjith and Marcel, S{'e}bastien and Ganapathi, Iyyakutti Iyappan and others},
  booktitle={2023 IEEE International Joint Conference on Biometrics (IJCB)},
  pages={1--10},
  year={2023},
  organization={IEEE},
  doi={10.1109/IJCB57857.2023.10449062}
}
```
