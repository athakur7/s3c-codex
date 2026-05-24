from __future__ import annotations

import cv2
import numpy as np


def _lab_l(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[:, :, 0]


def forward_branch_costs(image_rgb: np.ndarray, e_sh: np.ndarray, lambda_sh: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-energy branch costs with optional shadow discontinuity term."""
    l = _lab_l(image_rgb)
    left = np.roll(l, 1, axis=1)
    right = np.roll(l, -1, axis=1)
    up = np.roll(l, 1, axis=0)

    c_u = np.abs(right - left)
    c_l = c_u + np.abs(up - left)
    c_r = c_u + np.abs(up - right)

    if lambda_sh > 0:
        sh_left = np.roll(e_sh, 1, axis=1)
        sh_right = np.roll(e_sh, -1, axis=1)
        sh_up = np.roll(e_sh, 1, axis=0)
        s_u = np.abs(sh_right - sh_left)
        s_l = s_u + np.abs(sh_up - sh_left)
        s_r = s_u + np.abs(sh_up - sh_right)
        c_u = c_u + lambda_sh * s_u
        c_l = c_l + lambda_sh * s_l
        c_r = c_r + lambda_sh * s_r
    return c_l, c_u, c_r
