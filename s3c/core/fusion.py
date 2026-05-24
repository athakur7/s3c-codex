from __future__ import annotations

import cv2
import numpy as np

from s3c.utils import normalize_map


def baseline_fusion(e_g: np.ndarray, e_s: np.ndarray, e_sh: np.ndarray) -> np.ndarray:
    return normalize_map(e_g * e_s * e_sh)


def s3c_fusion(
    e_g: np.ndarray,
    e_s: np.ndarray,
    e_sh: np.ndarray,
    e_str: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    p_shadow_mult: float,
    p_struct_mult: float,
    tau_struct: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Implements § 2.3 dual-penalty fusion."""
    e_base = alpha * e_g + beta * e_s + gamma * e_sh + delta * e_str
    gx = cv2.Sobel(e_sh.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(e_sh.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    boundary = (np.sqrt(gx * gx + gy * gy) > 0.05).astype(np.float32)
    struct_hard = (e_str > tau_struct).astype(np.float32)
    m = float(np.max(e_base) + 1e-8)
    e = e_base + (p_shadow_mult * m) * boundary + (p_struct_mult * m) * struct_hard
    return normalize_map(e), e_base
