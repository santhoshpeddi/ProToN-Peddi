# Emeršič et al. (2023): UERC 2023 Participant (MEM-Ear)

This directory contains our implementation of a participant (MEM-Ear) baseline associated with:

> Žiga Emeršič, Tetsushi Ohki, Muku Akasaka, Takahiko Arakawa, Soshi Maeda, Masora Okano, Yuya Sato, Anjith George, Sébastien Marcel, Iyyakutti Iyappan Ganapathi, and others,  
> **The Unconstrained Ear Recognition Challenge 2023: Maximizing Performance and Minimizing Bias**,  
> *2023 IEEE International Joint Conference on Biometrics (IJCB)*, pp. 1-10, 2023.

## What is implemented here

The implementation in this folder provides:

- a clean training entry point via `main.py`;
- dataset loading and preparation in `dataset.py` and `utils.py`;
- model definitions in `model.py`;
- IResNet backbone in `iresnet.py`;
- loss functions in `loss.py`;
- training utilities in `train.py`;
- experiment defaults in `config.py`.

## Implementation scope

In our comparative study, we preserve the MEM-Ear design and re-run it under the unified evaluation protocol used in our paper.

To keep the comparison fair and reproducible:

- the architectural family and overall training strategy are kept aligned with the source paper;
- the baseline preserves the main components including the ConvNeXt-Tiny, EfficientNet-B3, and IResNet-100;
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
MEM-EAR[6]/
├── README.md
├── main.py
├── config.py
├── dataset.py
├── model.py
├── iresnet.py
├── loss.py
├── train.py
└── utils.py
```

## How to run

From this directory:

```bash
python main.py \
  --dataset-root /path/to/dataset \
  --out-dir /path/to/outputs
```

Use local pretrained checkpoints for the three branches:

```bash
python main.py \
  --dataset-root /path/to/dataset \
  --out-dir /path/to/outputs \
  --convnext-weights /path/to/convnext_tiny.pth \
  --efficientnet-weights /path/to/efficientnet_b3.pth \
  --iresnet-weights /path/to/r100_ms1mv2.pth
```

Change common training settings:

```bash
python main.py \
  --dataset-root /path/to/dataset \
  --out-dir /path/to/outputs \
  --epochs 150 \
  --batch-size 64 \
  --lr 0.001 \
  --weight-decay 0.001 \
  --fusion-weights 1.0 1.0 1.0
```

## Main command-line arguments

| Argument | Description |
|---|---|
| `--dataset-root` | Root directory of the dataset |
| `--out-dir` | Directory to save CSVs, checkpoints, metrics, and embeddings |
| `--train-folder` | Folder containing class-wise training images |
| `--epochs` | Number of training epochs |
| `--batch-size` | Batch size |
| `--lr` | Initial learning rate |
| `--weight-decay` | AdamW weight decay |
| `--val-ratio` | Fraction of training images used for internal validation |
| `--val-interval` | Validate every N epochs |
| `--num-workers` | Number of dataloader workers |
| `--device` | Device to use, such as `cuda` or `cpu` |
| `--embedding-dim` | Shared embedding dimensionality for all branches |
| `--dropout` | Dropout before the classification head |
| `--fusion-weights` | Three weights for ConvNeXt, EfficientNet, and IResNet embeddings |
| `--convnext-weights` | Local ConvNeXt-Tiny checkpoint |
| `--efficientnet-weights` | Local EfficientNet-B3 checkpoint |
| `--iresnet-weights` | Local IResNet-100 checkpoint |
| `--use-demographic-weights` | Enable demographic sample weighting during training |
| `--demographics-csv` | CSV containing `cid`, `gender`, and `ethnicity` columns |

## Output files

The script creates:

- `csv/train.csv` — generated training metadata;
- `csv/train_split.csv` — training split with closed-set integer labels;
- `csv/val_split.csv` — validation split with closed-set integer labels;
- `csv/label_mapping.json` — mapping from identity strings to training labels;
- `checkpoints/best_model.pth` — best checkpoint by validation loss;
- `checkpoints/last_model.pth` — final checkpoint at the end of training;
- `metrics/history.csv` — epoch-wise training and validation log;
- `metrics/summary.json` — compact run summary;

## Example output layout

```text
outputs/
├── csv/
│   ├── train.csv
│   ├── train_split.csv
│   ├── val_split.csv
│   └── label_mapping.json
├── checkpoints/
│   ├── best_model.pth
│   └── last_model.pth
└── metrics/
    ├── history.csv
    └── summary.json
```

## Reproducibility notes

- Set `--seed` for repeatable CSV splitting and training initialization.
- By default, the model does not download pretrained weights. Local checkpoints can be passed explicitly through the three branch-specific weight arguments.

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
