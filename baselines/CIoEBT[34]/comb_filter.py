from __future__ import annotations

import numpy as np


def comb_filter_fir(x: np.ndarray, L: int) -> np.ndarray:
    """FIR comb filter y(n) = x(n) - x(n-L)."""
    x = np.asarray(x, dtype=np.float64)
    y = x.copy()
    if L <= 0 or L >= len(x):
        return y
    y[L:] = x[L:] - x[:-L]
    y[:L] = x[:L]
    return y


def comb_filter_multi_pass(x: np.ndarray, order_L: int, passes: int = 3) -> np.ndarray:
    """Apply a comb filter repeatedly as a practical multi-pass band-comb implementation."""
    y = np.asarray(x, dtype=np.float64)
    for _ in range(max(1, int(passes))):
        y = comb_filter_fir(y, order_L)
    return y


def to_binary_template(z: np.ndarray) -> np.ndarray:
    """Convert a real-valued vector into a non-invertible binary EarCode."""
    z = np.asarray(z, dtype=np.float64)
    z = z - np.mean(z)
    return (z >= 0).astype(np.uint8)


def protect_with_comb_and_expand(
    feats: np.ndarray,
    order_L: int,
    aug_factor: int = 10,
    passes: int = 3,
    make_binary: bool = True,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    feats = np.asarray(feats, dtype=np.float64)

    out = []
    for i in range(len(feats)):
        x = feats[i]
        for t in range(aug_factor):
            L = int(max(1, min(len(x) - 1, order_L + (t % 3) - 1)))
            p = int(max(1, passes + (t % 2)))
            y = comb_filter_multi_pass(x, order_L=L, passes=p)
            y = y + 1e-9 * rng.standard_normal(size=y.shape)
            if make_binary:
                y = to_binary_template(y)
            out.append(y)
    return np.stack(out, axis=0)


def expand_labels(labels: np.ndarray, aug_factor: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    return np.repeat(labels, repeats=aug_factor, axis=0)
