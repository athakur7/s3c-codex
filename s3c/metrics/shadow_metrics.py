from __future__ import annotations

import numpy as np
from skimage.transform import resize


def shadow_iou(original_mask: np.ndarray | None, carved_mask: np.ndarray, target_shape: tuple[int, int]) -> float:
    if original_mask is None:
        return float("nan")
    resized = resize(original_mask.astype(np.float32), target_shape, order=1, preserve_range=True, anti_aliasing=True)
    a = resized > 0.5
    b = carved_mask > 0.5
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / (union + 1e-8))


def shadow_preservation_ratio(original_mask: np.ndarray | None, carved_mask: np.ndarray, scale_factor: float) -> float:
    if original_mask is None:
        return float("nan")
    num = float((carved_mask > 0.5).sum())
    den = float((original_mask > 0.5).sum() * scale_factor + 1e-8)
    return num / den
