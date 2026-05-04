# Omara and Soliman (2025): CIoEBT with CaffeNet: Cancelable Internet of Ear Biometric Things

This directory contains our implementation of the baseline derived from:

> Ibrahim Omara and Randa F. Soliman,  
> **CIoEBT: Cancelable Internet of Ear Biometric Things Based -- A Novel Deep Metric Learning Approach**,  
> *Expert Systems with Applications*, vol. 297, article 129439, 2025.

## What is implemented here

The implementation in this folder provides:

- a clean training entry point via `main.py`;
- dataset loading and CaffeNet preprocessing in `dataset.py`;
- model definitions in `model.py`;
- DEP-ML metric learning in `dep_ml.py`;
- comb-filter EarCode generation in `comb_filter.py`;
- training utilities in `train.py`;
- experiment defaults in `config.py`.

## Implementation scope

In our comparative study, we preserve the CIoEBT design and re-run it under the unified evaluation protocol used in our paper.

To keep the comparison fair and reproducible:

- the architectural family and overall training strategy are kept aligned with the source paper;
- explicitly available settings are preserved where possible;
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

`main.py` creates an internal class-consistent train/eval split from the images inside each class folder using `eval_ratio`.

## External Caffe files required

This baseline uses the original CaffeNet model through `caffemodel2pytorch`. You must download the required pretrained weights and generate the necessary protocol buffer files.

You can download the pretrained CaffeNet weights by following the official BVLC Caffe instructions and running the script they provided:

```bash
python caffe/scripts/download_model_binary.py
```

After downloading, verify that the model file exists at the following path:

```text
caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
```

To compile the Caffe protocol definitions, run the following command:

```bash
python -m grpc_tools.protoc \
   -I caffe/src/caffe/proto \
   --python_out caffe/src/caffe/proto \
   caffe/src/caffe/proto/caffe.proto
```

## Files in this folder

```text
CIoEBT[34]/
├── README.md
├── main.py
├── config.py
├── dataset.py
├── model.py
├── train.py
├── dep_ml.py
└── comb_filter.py
```

## How to run

From this directory:

```bash
python main.py \
  --dataset-root /path/to/dataset \
  --out-dir /path/to/outputs \
  --caffe-deploy /path/to/caffe/models/bvlc_reference_caffenet/deploy.prototxt \
  --caffe-weights /path/to/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
  --caffe-proto /path/to/caffe/src/caffe/proto/caffe.proto
```

## Main command-line arguments

| Argument | Description |
|---|---|
| `--dataset-root` | Root directory of the dataset |
| `--train-folder` | Folder containing class-wise training images |
| `--out-dir` | Directory to save checkpoints, metrics, and DEP-ML outputs |
| `--caffe-deploy` | Path to CaffeNet `deploy.prototxt` |
| `--caffe-weights` | Path to CaffeNet `.caffemodel` weights |
| `--caffe-proto` | Path to local `caffe.proto` |
| `--caffenet-checkpoint` | Optional PyTorch checkpoint for the converted CaffeNet model |
| `--skip-caffenet-training` | Skip supervised fine-tuning and only extract CaffeNet features |
| `--epochs` | Number of CaffeNet fine-tuning epochs |
| `--batch-size` | Batch size |
| `--lr` | Learning rate for CaffeNet fine-tuning |
| `--weight-decay` | Weight decay for CaffeNet fine-tuning |
| `--eval-ratio` | Fraction of images per class used for internal evaluation |
| `--depml-u` | DEP-ML upper bound for similar-pair distance |
| `--depml-l` | DEP-ML lower bound for dissimilar-pair distance |
| `--depml-cycles` | Number of DEP-ML constraint-refresh cycles |
| `--depml-eta` | DEP-ML convergence threshold |
| `--depml-gamma` | DEP-ML shaping/regularization parameter |
| `--knn-k` | Number of neighbors for final DEP-ML k-NN classification |
| `--comb-orders` | Comb-filter order or list of orders, e.g. `6 8 10 12` |
| `--comb-aug-factor` | Template expansion factor for comb-filter protection |
| `--comb-passes` | Number of comb-filter passes |
| `--no-comb-protection` | Fit DEP-ML directly on CaffeNet features |
| `--save-feature-arrays` | Save extracted/intermediate feature arrays |
| `--from-feature-arrays` | Run DEP-ML from existing `.npy` feature arrays |

## Output files

The script creates:

- `caffenet_best.pth` — best CaffeNet checkpoint by internal evaluation accuracy;
- `caffenet_last.pth` — final CaffeNet checkpoint;
- `training_history.csv` — epoch-wise CaffeNet training log;
- `run_summary.json` — compact summary of the full run;
- `comb_order_<L>/depml_model_state.npz` — learned DEP-ML metric and stored training features for comb-filter order `L`;
- `comb_order_<L>/depml_summary.json` — DEP-ML evaluation summary for comb-filter order `L`;
- optional `.npy` feature arrays when `--save-feature-arrays` is used.

## Example output layout

```text
outputs/
├── caffenet_best.pth
├── caffenet_last.pth
├── training_history.csv
├── run_summary.json
├── comb_order_6/
│   ├── depml_model_state.npz
│   └── depml_summary.json
├── comb_order_8/
│   ├── depml_model_state.npz
│   └── depml_summary.json
└── ...
```

## Reproducibility notes

- Set `--seed` for deterministic splitting and repeatable experiments.
- The default comb-filter orders are `6`, `8`, `10`, and `12`.
- Some operational details are implementation-level choices for reproducible execution when the original paper does not specify every release-ready detail.

## Citation

```bibtex
@article{omara2025cioebt,
  title     = {CIoEBT: cancelable internet of ear biometric things based--a novel deep metric learning approach},
  author    = {Omara, Ibrahim and Soliman, Randa F},
  journal   = {Expert Systems with Applications},
  volume    = {297},
  pages     = {129439},
  year      = {2025},
  publisher = {Elsevier},
  doi       = {10.1016/j.eswa.2025.129439}
}
```
