from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from s3c.utils import normalize_map

logger = logging.getLogger(__name__)


def shadow_map_gt(mask_path: Path, shape: tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    """Load GT mask and feather edges for soft E_sh."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    m = (mask > 127).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return normalize_map(m)


def shadow_map_baseline_ycbcr(image_rgb: np.ndarray) -> np.ndarray:
    """Hashemzadeh-like heuristic shadow detector in YCrCb space."""
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y = ycrcb[:, :, 0]
    mean_global = float(y.mean())
    shadow = np.zeros_like(y, dtype=np.uint8)
    for b in range(81, 2, -16):
        k = max(3, b if b % 2 == 1 else b + 1)
        local_mean = cv2.blur(y, (k, k))
        cond = (y < 0.6 * mean_global) | (y < 0.7 * local_mean)
        shadow = np.maximum(shadow, cond.astype(np.uint8))
    shadow = cv2.medianBlur((shadow * 255).astype(np.uint8), 5).astype(np.float32) / 255.0
    return normalize_map(shadow)


def shadow_map_auto(image_rgb: np.ndarray, tau_r: float = 0.65, tau_c: float = 0.12, window: int = 51) -> np.ndarray:
    """Improved unsupervised detector using Lab local illumination and chroma divergence."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0] * (100.0 / 255.0)
    a = (lab[:, :, 1] - 128.0) / 127.0
    b = (lab[:, :, 2] - 128.0) / 127.0

    l_norm = l / 100.0
    k = window if window % 2 == 1 else window + 1
    mean_l = cv2.blur(l_norm, (k, k))
    mean_a = cv2.blur(a, (k, k))
    mean_b = cv2.blur(b, (k, k))

    eps = 1e-6
    r = (l_norm + eps) / (mean_l + eps)
    chroma_diff = np.sqrt((a - mean_a) ** 2 + (b - mean_b) ** 2)
    candidate = (r < tau_r) & (chroma_diff < tau_c)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    min_area = int(0.001 * image_rgb.shape[0] * image_rgb.shape[1])
    keep = np.zeros_like(cleaned, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= max(1, min_area):
            keep[labels == i] = 1

    soft = ndimage.gaussian_filter(keep.astype(np.float32), sigma=1.0)
    return normalize_map(soft)
