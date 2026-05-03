from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np

from config import ShapeAutoencoderConfig, DEFAULT_CONFIG


@dataclass
class ExtractionParams:
    blur_kernel: Tuple[int, int] = DEFAULT_CONFIG.gauss_blur_kernel
    canny_min: int = DEFAULT_CONFIG.canny_min_thresh
    canny_max: int = DEFAULT_CONFIG.canny_max_thresh
    max_points: int = DEFAULT_CONFIG.num_feature_points
    gftt_quality: float = DEFAULT_CONFIG.gftt_quality_level
    gftt_min_distance: float = DEFAULT_CONFIG.gftt_min_distance


def extraction_params_from_config(cfg: ShapeAutoencoderConfig) -> ExtractionParams:
    return ExtractionParams(
        blur_kernel=cfg.gauss_blur_kernel,
        canny_min=cfg.canny_min_thresh,
        canny_max=cfg.canny_max_thresh,
        max_points=cfg.num_feature_points,
        gftt_quality=cfg.gftt_quality_level,
        gftt_min_distance=cfg.gftt_min_distance,
    )


def to_grayscale_by_average(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr.copy()
    gray = img_bgr.astype(np.float32).mean(axis=2)
    return np.clip(gray, 0, 255).astype(np.uint8)


def gaussian_blur(gray: np.ndarray, kernel: Tuple[int, int]) -> np.ndarray:
    kx, ky = kernel
    if kx % 2 == 0:
        kx += 1
    if ky % 2 == 0:
        ky += 1
    return cv2.GaussianBlur(gray, (kx, ky), sigmaX=0)


def canny_edges(blurred_gray: np.ndarray, tmin: int, tmax: int) -> np.ndarray:
    return cv2.Canny(blurred_gray, tmin, tmax)


def largest_contour(edges: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def contour_centroid(contour: np.ndarray) -> Tuple[float, float]:
    m = cv2.moments(contour)
    if abs(m['m00']) < 1e-9:
        pts = contour.reshape(-1, 2).astype(np.float32)
        return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    return float(m['m10'] / m['m00']), float(m['m01'] / m['m00'])


def fit_ellipse_center(contour: Optional[np.ndarray], fallback_hw: Tuple[int, int]) -> Tuple[float, float]:
    if contour is None:
        h, w = fallback_hw
        return w / 2.0, h / 2.0
    if len(contour) < 5:
        return contour_centroid(contour)
    try:
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), _, _ = ellipse
        return float(cx), float(cy)
    except cv2.error:
        return contour_centroid(contour)


def contour_mask(shape_hw: Tuple[int, int], contour: Optional[np.ndarray]) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=np.uint8)
    if contour is not None:
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    else:
        mask[:, :] = 255
    return mask


def extract_strong_points(blurred_gray: np.ndarray, mask: np.ndarray, params: ExtractionParams) -> np.ndarray:
    pts: List[Tuple[float, float]] = []
    corners = cv2.goodFeaturesToTrack(
        blurred_gray,
        maxCorners=params.max_points,
        qualityLevel=params.gftt_quality,
        minDistance=params.gftt_min_distance,
        mask=mask,
        blockSize=3,
        useHarrisDetector=False,
    )
    if corners is not None:
        corners = corners.reshape(-1, 2)
        pts.extend([(float(x), float(y)) for x, y in corners])

    if len(pts) < params.max_points:
        fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        kps = fast.detect(blurred_gray, mask)
        kps = sorted(kps, key=lambda k: k.response, reverse=True)
        for kp in kps:
            if len(pts) >= params.max_points:
                break
            pts.append((float(kp.pt[0]), float(kp.pt[1])))

    return np.array(pts, dtype=np.float32)


def extract_distance_features(img_bgr: np.ndarray, params: ExtractionParams, debug: bool = False):
    gray = to_grayscale_by_average(img_bgr)
    blurred = gaussian_blur(gray, params.blur_kernel)
    edges = canny_edges(blurred, params.canny_min, params.canny_max)
    contour = largest_contour(edges)

    cx, cy = fit_ellipse_center(contour, gray.shape)
    mask = contour_mask(gray.shape, contour)
    pts = extract_strong_points(blurred, mask, params)

    if pts.size == 0:
        dists = np.zeros((0,), dtype=np.float32)
    else:
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        dists = np.sqrt(dx * dx + dy * dy).astype(np.float32)

    dists = np.sort(dists)
    if len(dists) < params.max_points:
        dists = np.pad(dists, (0, params.max_points - len(dists)), constant_values=0.0)
    else:
        dists = dists[: params.max_points]

    dbg = None
    if debug:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if contour is not None:
            cv2.drawContours(vis, [contour], -1, (255, 0, 0), 2)
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(vis, ellipse, (0, 255, 0), 2)
                except cv2.error:
                    pass
        cv2.circle(vis, (int(round(cx)), int(round(cy))), 4, (0, 0, 255), -1)
        for (x, y) in pts[: params.max_points]:
            cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 255, 255), -1)
        dbg = {'gray': gray, 'blurred': blurred, 'edges': edges, 'mask': mask, 'vis': vis}

    return dists.astype(np.float32), dbg


def aug_random_noise(img_bgr: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, img_bgr.shape).astype(np.float32)
    out = img_bgr.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_contrast(img_bgr: np.ndarray, alpha: float = 1.35, beta: float = 0.0) -> np.ndarray:
    out = img_bgr.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_gamma(img_bgr: np.ndarray, gamma: float = 1.35) -> np.ndarray:
    inv = 1.0 / max(gamma, 1e-6)
    table = (np.linspace(0, 1, 256) ** inv) * 255.0
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(img_bgr, table)
