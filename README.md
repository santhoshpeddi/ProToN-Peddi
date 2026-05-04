# Ear Recognition Baselines for ProtoN

This repository contains the baseline implementations used in our comparative study for **ProtoN: Prototype Node Graph Neural Network for Unconstrained Multi-Impression Ear Recognition**.

>**Paper:** *ProtoN: Prototype Node Graph Neural Network for Unconstrained Multi-Impression Ear Recognition*  
>Santhoshkumar Peddi, Sadhvik Bathini, Arun Balasubramanian, Monalisa Sarma, and Debasis Samanta  
>ArXiv: 2508.04381, 2025.

## Purpose of this repository

Public implementations of ear-recognition methods are often incomplete, tied to different datasets, or evaluated under dataset splits that are not directly comparable. To support transparent peer review and reproducibility, we re-implemented the baseline methods used in our paper and re-ran them under a unified experimental protocol.

In particular:

- each baseline is placed in its own directory;
- each baseline directory contains a dedicated `main.py` entry point and a baseline-specific `README.md`;
- the implementations follow the original papers as closely as possible;
- when a paper does not fully specify a training detail, or when its original data split is incompatible with our evaluation setup, we preserve the published method design and adapt the experimental protocol only as needed for a fair comparison.

## Repository layout

```text
.
├── README.md
├── requirements.txt
└── baselines/
    └── ExplainableEar[23]/
    └── DenseNet[4]/
    └── PrivacyPreserving[5]/
    └── MDFNet[29]/
    └── ViTEar[6]/
    └── MEM-EAR[6]/
    └── Ensemble[26]/
    └── Autoencoder[33]/
    └── CIoEBT[34]/
    ...
```

## Implemented baselines

| Baseline | Source paper | DOI |
|---|---|---|
| Alshazly et al. (2021) | *Towards Explainable Ear Recognition Systems Using Deep Residual Networks* | [Link](https://doi.org/10.1109/ACCESS.2021.3109441) |
| El-Naggar and Bourlai (2022) | *Exploring Deep Learning Ear Recognition in Thermal Images* | [Link](https://doi.org/10.1109/TBIOM.2022.3218151) |
| Chowdhury et al. (2022) | *Privacy Preserving Ear Recognition System Using Transfer Learning in Industry 4.0* | [Link](https://doi.org/10.1109/TII.2022.3196343) |
| Aiadi et al. (2023) | *MDFNet: an unsupervised lightweight network for ear print recognition* | [Link](https://doi.org/10.1007/s12652-022-04028-z) |
| Emeršič et al. (2023) | *The Unconstrained Ear Recognition Challenge 2023: Maximizing Performance and Minimizing Bias* | [Link](https://doi.org/10.1109/IJCB57857.2023.10449062) |
| Mehta et al. (2024) | *Ensemble-based hybrid transfer approach for an effective 2D ear recognition system* | [Link](https://doi.org/10.1109/ACCESS.2024.3485514) |
| Pal et al. (2025) | *Exploration of a shape-focused autoencoder for improved ear biometrics* | [Link](https://doi.org/10.1007/s11042-025-20905-z) |
| Omara and Soliman (2025) | *CIoEBT: Cancelable Internet of Ear Biometric Things Based -- A Novel Deep Metric Learning Approach* | [Link](https://doi.org/10.1016/j.eswa.2025.129439) |

## Reproducibility statement

The goal of this repository is **fair and transparent comparison**, not to claim that all originally reported numbers can be reproduced bit-for-bit on different datasets. For each baseline, we:

1. follow the published architecture and training strategy as closely as possible;
2. preserve explicitly reported hyperparameters wherever available;
3. document any dataset-specific or protocol-specific adjustments required to align the evaluation with our paper.

Please consult each baseline-level `README.md` for method-specific details.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running a baseline

Example:

```bash
cd baselines/ExplainableEar[23]
python main.py --dataset-root /path/to/dataset --train-all
```

## Citation

If you use this repository, please cite both the original baseline paper and our paper.

### Our paper

```bibtex
@article{peddi2025proton,
  title   = {ProtoN: Prototype Node Graph Neural Network for Unconstrained Multi-Impression Ear Recognition},
  author  = {Peddi, Santhoshkumar and Bathini, Sadhvik and Balasubramanian, Arun and Sarma, Monalisa and Samanta, Debasis},
  journal = {arXiv preprint arXiv:2508.04381},
  year    = {2025}
}
```

## Notes for reviewers

This repository is organized to make review straightforward:

- every baseline is isolated and self-documented;
- the training entry point is explicit;
- baseline-specific assumptions are written down close to the code;
- implementation choices introduced for fair comparison are disclosed rather than hidden.
