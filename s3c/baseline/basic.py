from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from s3c.carve.dp import backtrack_min_seam, cumulative_forward_cost
from s3c.carve.remove import overlay_seam, remove_seam_2d, remove_seam_3d, smooth_seam_2point
from s3c.core.gradient import gradient_map_lab_l


@dataclass(slots=True)
class CarveResult:
    image: np.ndarray
    mask: np.ndarray
    importance: np.ndarray
    seam_overlay: np.ndarray
    seam_through_shadow_pct: float
    runtime_sec: float


def run_basic(image_rgb: np.ndarray, target_w: int, initial_mask: np.ndarray | None) -> CarveResult:
    """Classic seam carving using backward energy (gradient-only)."""
    t0 = time.perf_counter()
    work_img = image_rgb.copy()
    work_mask = initial_mask.copy() if initial_mask is not None else np.zeros(work_img.shape[:2], dtype=np.float32)

    seam_hits, seam_total = 0, 0
    overlay = work_img.copy()
    importance = None
    while work_img.shape[1] > target_w:
        energy = gradient_map_lab_l(work_img)
        if importance is None:
            importance = energy.copy()
        zeros = np.zeros_like(energy, dtype=np.float32)
        cum, bt = cumulative_forward_cost(energy, zeros, zeros, zeros)
        seam = backtrack_min_seam(cum, bt)
        seam_hits += int(np.sum(work_mask[np.arange(work_mask.shape[0]), seam] > 0.5))
        seam_total += int(work_mask.shape[0])
        overlay = overlay_seam(overlay, seam)
        work_img = smooth_seam_2point(work_img, seam)
        work_img = remove_seam_3d(work_img, seam)
        work_mask = remove_seam_2d(work_mask, seam)

    pct = (100.0 * seam_hits / max(1, seam_total))
    return CarveResult(
        image=work_img,
        mask=work_mask,
        importance=importance if importance is not None else np.zeros(work_img.shape[:2], dtype=np.float32),
        seam_overlay=overlay,
        seam_through_shadow_pct=pct,
        runtime_sec=time.perf_counter() - t0,
    )
