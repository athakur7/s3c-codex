from __future__ import annotations

import numpy as np


def _seam_mask(shape: tuple[int, int], seam: np.ndarray) -> np.ndarray:
    h, w = shape
    m = np.ones((h, w), dtype=bool)
    m[np.arange(h), seam] = False
    return m


def remove_seam_2d(arr: np.ndarray, seam: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    mask = _seam_mask((h, w), seam)
    return arr[mask].reshape(h, w - 1)


def remove_seam_3d(arr: np.ndarray, seam: np.ndarray) -> np.ndarray:
    h, w, c = arr.shape
    mask = _seam_mask((h, w), seam)
    return arr[mask].reshape(h, w - 1, c)


def smooth_seam_2point(image_rgb: np.ndarray, seam: np.ndarray) -> np.ndarray:
    out = image_rgb.copy().astype(np.float32)
    h, w, _ = out.shape
    for i in range(h):
        j = seam[i]
        if 0 < j < w - 1:
            out[i, j] = 0.5 * out[i, j - 1] + 0.5 * out[i, j + 1]
    return out.astype(np.uint8)


def smooth_seam_bilateral_1d(
    image_rgb: np.ndarray, seam: np.ndarray, window: int = 5, sigma_spatial: float = 1.5, sigma_range: float = 15.0
) -> np.ndarray:
    out = image_rgb.copy().astype(np.float32)
    h, w, _ = out.shape
    r = window // 2
    for i in range(h):
        j = seam[i]
        lo, hi = max(0, j - r), min(w - 1, j + r)
        vals = out[i, lo : hi + 1]
        coords = np.arange(lo, hi + 1)
        ws = np.exp(-0.5 * ((coords - j) / max(sigma_spatial, 1e-6)) ** 2)[:, None]
        wr = np.exp(-0.5 * ((vals - out[i, j]) / max(sigma_range, 1e-6)) ** 2)
        w_all = ws * wr
        out[i, j] = np.sum(w_all * vals, axis=0) / (np.sum(w_all, axis=0) + 1e-6)
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_seam(image_rgb: np.ndarray, seam: np.ndarray, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    out = image_rgb.copy()
    out[np.arange(out.shape[0]), seam] = np.array(color, dtype=np.uint8)
    return out
