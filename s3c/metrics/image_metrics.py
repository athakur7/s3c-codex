from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize


def ssim_vs_rescaled(orig_rgb: np.ndarray, carved_rgb: np.ndarray) -> float:
    ref = resize(orig_rgb, carved_rgb.shape[:2], order=1, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    return float(structural_similarity(ref, carved_rgb, channel_axis=2, data_range=255))


def psnr_vs_rescaled(orig_rgb: np.ndarray, carved_rgb: np.ndarray) -> float:
    ref = resize(orig_rgb, carved_rgb.shape[:2], order=1, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    return float(peak_signal_noise_ratio(ref, carved_rgb, data_range=255))
