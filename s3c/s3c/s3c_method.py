from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from s3c.carve.dp import backtrack_min_seam, cumulative_forward_cost
from s3c.carve.forward_energy import forward_branch_costs
from s3c.carve.remove import overlay_seam, remove_seam_2d, remove_seam_3d, smooth_seam_bilateral_1d
from s3c.config import S3CConfig
from s3c.core.fusion import s3c_fusion
from s3c.core.gradient import gradient_map_lab_l
from s3c.core.saliency import cluster_saliency_map
from s3c.core.shadow import shadow_map_auto, shadow_map_gt
from s3c.core.structure import structure_map


@dataclass(slots=True)
class CarveResult:
    image: np.ndarray
    mask: np.ndarray
    importance: np.ndarray
    seam_overlay: np.ndarray
    seam_through_shadow_pct: float
    runtime_sec: float
    used_shadow_mode: str


def run_s3c(
    image_rgb: np.ndarray, target_w: int, config: S3CConfig, mask_path: Path | None, initial_mask: np.ndarray | None
) -> CarveResult:
    t0 = time.perf_counter()
    work_img = image_rgb.copy()
    e_s = cluster_saliency_map(work_img, k=6)
    e_str = structure_map(work_img, sigma=config.sigma_tensor, w_line=config.w_line)
    used = config.shadow_mode
    if config.shadow_mode == "gt" and mask_path is not None:
        e_sh = shadow_map_gt(mask_path, shape=work_img.shape[:2], sigma=1.0)
    else:
        used = "auto"
        e_sh = shadow_map_auto(work_img, tau_r=config.tau_r, tau_c=config.tau_c, window=config.local_window)
    work_mask = initial_mask.copy() if initial_mask is not None else (e_sh > 0.5).astype(np.float32)

    seam_hits, seam_total = 0, 0
    overlay = work_img.copy()
    importance = None
    while work_img.shape[1] > target_w:
        e_g = gradient_map_lab_l(work_img)
        energy, _ = s3c_fusion(
            e_g=e_g,
            e_s=e_s,
            e_sh=e_sh,
            e_str=e_str,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            delta=config.delta,
            p_shadow_mult=config.p_shadow,
            p_struct_mult=config.p_struct,
            tau_struct=config.tau_struct,
        )
        if importance is None:
            importance = energy.copy()
        c_l, c_u, c_r = forward_branch_costs(work_img, e_sh=e_sh, lambda_sh=config.lambda_sh)
        cum, bt = cumulative_forward_cost(energy, c_l, c_u, c_r)
        seam = backtrack_min_seam(cum, bt)
        seam_hits += int(np.sum(work_mask[np.arange(work_mask.shape[0]), seam] > 0.5))
        seam_total += int(work_mask.shape[0])
        overlay = overlay_seam(overlay, seam)
        work_img = smooth_seam_bilateral_1d(
            work_img,
            seam,
            window=config.bilateral_window,
            sigma_spatial=config.bilateral_sigma_spatial,
            sigma_range=config.bilateral_sigma_range,
        )
        work_img = remove_seam_3d(work_img, seam)
        e_s = remove_seam_2d(e_s, seam)
        e_sh = remove_seam_2d(e_sh, seam)
        e_str = remove_seam_2d(e_str, seam)
        work_mask = remove_seam_2d(work_mask, seam)

    pct = (100.0 * seam_hits / max(1, seam_total))
    return CarveResult(
        image=work_img,
        mask=work_mask,
        importance=importance if importance is not None else np.zeros(work_img.shape[:2], dtype=np.float32),
        seam_overlay=overlay,
        seam_through_shadow_pct=pct,
        runtime_sec=time.perf_counter() - t0,
        used_shadow_mode=used,
    )
