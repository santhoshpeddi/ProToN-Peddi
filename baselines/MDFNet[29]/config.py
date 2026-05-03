"""Configuration for the Aiadi et al. MDFNet baseline."""

from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path("data/uerc23_dataset")
OUT_DIR = Path("outputs")
TRAIN_FOLDER = "train"

# MDFNet paper subset definitions. Table-2 style parameters:
# M = number of PCA filters, p = filter size, block = block size.
SUBSETS = {
    1: {"num_filters": 5, "filter_size": 5, "block_size": 16},
    2: {"num_filters": 5, "filter_size": 7, "block_size": 32},
    3: {"num_filters": 9, "filter_size": 9, "block_size": 16},
    4: {"num_filters": 9, "filter_size": 9, "block_size": 100},
    5: {"num_filters": 9, "filter_size": 9, "block_size": 25},
    6: {"num_filters": 7, "filter_size": 7, "block_size": 100},
}

MDFNET_SUBSET = 5
BETA = 0.5
DIRECTION_BINS = 8
IMAGE_SIZE = (96, 96)

# Data loading defaults
EVAL_RATIO = 0.20
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 2

# Linear SVM defaults
SVM_C = 1.0
SVM_MAX_ITER = 10000
