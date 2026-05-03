"""MDFNet feature extractor.

This implementation follows the MDFNet pipeline at a practical reproduction level:
PCA filter learning, learned-filter convolution, gradient magnitude/direction maps,
binary hashing, block-wise dual histograms, and power-L2 normalization.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def power_l2_normalize(hist: np.ndarray, beta: float = 0.5, eps: float = 1e-12) -> np.ndarray:
    h = np.sign(hist) * (np.abs(hist) ** beta)
    return h / (np.linalg.norm(h) + eps)


def direction_quantize(angle: np.ndarray, bins: int = 8) -> np.ndarray:
    q = np.floor((angle + np.pi) * (bins / (2.0 * np.pi))).astype(np.int32)
    return np.clip(q, 0, bins - 1)


class MDFNet:
    """
    Unsupervised MDFNet descriptor extractor.

    Parameters
    ----------
    num_filters:
        Number of PCA filters, denoted M in the paper.
    filter_size:
        Square PCA filter size.
    block_size:
        Spatial block size for histogram extraction.
    beta:
        Power normalization exponent.
    direction_bins:
        Number of quantized gradient direction bins.
    device:
        Torch device used for convolution operations.
    """

    def __init__(
        self,
        num_filters: int = 9,
        filter_size: int = 9,
        block_size: int = 25,
        beta: float = 0.5,
        direction_bins: int = 8,
        device: str = "cpu",
    ):
        self.M = int(num_filters)
        self.p = int(filter_size)
        self.block = int(block_size)
        self.beta = float(beta)
        self.dir_bins = int(direction_bins)
        self.device = str(device)
        self.filters_: np.ndarray | None = None

    def fit(self, train_images: np.ndarray) -> None:
        """Learn PCA filters from grayscale training images of shape (N,H,W)."""
        if train_images.ndim != 3:
            raise ValueError("train_images must have shape (N,H,W)")
        if train_images.shape[1] < self.p or train_images.shape[2] < self.p:
            raise ValueError("filter_size must be smaller than the image height and width")

        p = self.p
        d = p * p
        cov = np.zeros((d, d), dtype=np.float64)

        for image in train_images:
            img = image.astype(np.float64, copy=False)
            windows = np.lib.stride_tricks.sliding_window_view(img, (p, p))
            patches = windows.reshape(-1, d)
            patches = patches - patches.mean(axis=1, keepdims=True)
            cov += patches.T @ patches

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]
        self.filters_ = eigvecs[:, : self.M].T.reshape(self.M, p, p).astype(np.float32)

    def _convolve_with_pca_filters(self, image: np.ndarray) -> np.ndarray:
        if self.filters_ is None:
            raise RuntimeError("Call fit() before transform().")
        filters = torch.from_numpy(self.filters_).unsqueeze(1).to(self.device)
        x = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        pad = self.p // 2
        x = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)
        y = F.conv2d(x, filters)
        return y.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def _compute_gradients(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        sh = torch.tensor([[-1.0, 0.0, 1.0]], device=self.device).view(1, 1, 1, 3)
        sv = torch.tensor([[-1.0], [0.0], [1.0]], device=self.device).view(1, 1, 3, 1)
        gh = F.conv2d(F.pad(x, (1, 1, 0, 0), mode="constant", value=0.0), sh)
        gv = F.conv2d(F.pad(x, (0, 0, 1, 1), mode="constant", value=0.0), sv)
        gh_np = gh.squeeze().detach().cpu().numpy()
        gv_np = gv.squeeze().detach().cpu().numpy()
        magnitude = np.sqrt(gv_np * gv_np + gh_np * gh_np).astype(np.float32)
        direction = np.arctan2(gv_np, gh_np).astype(np.float32)
        return magnitude, direction_quantize(direction, bins=self.dir_bins)

    def _binary_hash(self, responses: np.ndarray) -> np.ndarray:
        if responses.shape[0] != self.M:
            raise ValueError("Unexpected number of filter responses")
        bits = (responses > 0).astype(np.int32)
        hashed = np.zeros(responses.shape[1:], dtype=np.int32)
        for idx in range(self.M):
            hashed += (1 << idx) * bits[idx]
        return hashed

    def _blockwise_dual_hist(self, magnitude: np.ndarray, direction: np.ndarray, hashed: np.ndarray) -> np.ndarray:
        height, width = hashed.shape
        block = min(self.block, height, width)
        bins = 1 << self.M
        nby = height // block
        nbx = width // block
        features = []

        for by in range(nby):
            for bx in range(nbx):
                y0, y1 = by * block, (by + 1) * block
                x0, x1 = bx * block, (bx + 1) * block
                h = hashed[y0:y1, x0:x1].reshape(-1)
                mag = magnitude[y0:y1, x0:x1].reshape(-1).astype(np.float32)
                dr = direction[y0:y1, x0:x1].reshape(-1).astype(np.float32)
                hist_m = np.bincount(h, weights=mag, minlength=bins).astype(np.float32)
                hist_d = np.bincount(h, weights=dr, minlength=bins).astype(np.float32)
                features.append(
                    np.concatenate(
                        [
                            power_l2_normalize(hist_m, beta=self.beta),
                            power_l2_normalize(hist_d, beta=self.beta),
                        ],
                        axis=0,
                    )
                )

        if not features:
            raise RuntimeError("No histogram blocks were produced. Check image_size and block_size.")
        return np.concatenate(features, axis=0).astype(np.float32)

    def transform(self, images: np.ndarray) -> np.ndarray:
        """Extract MDFNet features from grayscale images of shape (N,H,W)."""
        if self.filters_ is None:
            raise RuntimeError("Call fit() before transform().")
        if images.ndim != 3:
            raise ValueError("images must have shape (N,H,W)")

        features = []
        for image in images:
            responses = self._convolve_with_pca_filters(image)
            magnitude, direction = self._compute_gradients(image)
            hashed = self._binary_hash(responses)
            features.append(self._blockwise_dual_hist(magnitude, direction, hashed))
        return np.stack(features, axis=0).astype(np.float32)

    def fit_transform(self, train_images: np.ndarray) -> np.ndarray:
        self.fit(train_images)
        return self.transform(train_images)

    def save(self, path: str) -> None:
        if self.filters_ is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        np.savez_compressed(
            path,
            filters=self.filters_,
            M=self.M,
            p=self.p,
            block=self.block,
            beta=self.beta,
            dir_bins=self.dir_bins,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "MDFNet":
        data = np.load(path, allow_pickle=False)
        obj = cls(
            num_filters=int(data["M"]),
            filter_size=int(data["p"]),
            block_size=int(data["block"]),
            beta=float(data["beta"]),
            direction_bins=int(data["dir_bins"]),
            device=device,
        )
        obj.filters_ = data["filters"].astype(np.float32)
        return obj
