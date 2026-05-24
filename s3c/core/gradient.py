from __future__ import annotations

import cv2
import numpy as np

from s3c.utils import normalize_map


def gradient_map_lab_l(image_rgb: np.ndarray) -> np.ndarray:
    """Compute E_g using Sobel gradients on Lab L channel."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0]
    gx = cv2.Sobel(l, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l, cv2.CV_32F, 0, 1, ksize=3)
    return normalize_map(np.abs(gx) + np.abs(gy))
