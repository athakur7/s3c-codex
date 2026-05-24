from __future__ import annotations

import cv2
import numpy as np
from scipy.stats import pearsonr
from skimage.transform import resize


def _cell_edge_density(edges: np.ndarray, cells: int = 16) -> np.ndarray:
    h, w = edges.shape
    ys = np.linspace(0, h, cells + 1, dtype=int)
    xs = np.linspace(0, w, cells + 1, dtype=int)
    out = []
    for i in range(cells):
        for j in range(cells):
            patch = edges[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            out.append(float((patch > 0).mean()) if patch.size else 0.0)
    return np.asarray(out, dtype=np.float32)


def structure_edge_corr(orig_rgb: np.ndarray, carved_rgb: np.ndarray) -> float:
    gt = resize(orig_rgb, carved_rgb.shape[:2], order=1, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    e1 = cv2.Canny(cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY), 80, 160)
    e2 = cv2.Canny(cv2.cvtColor(carved_rgb, cv2.COLOR_RGB2GRAY), 80, 160)
    d1 = _cell_edge_density(e1)
    d2 = _cell_edge_density(e2)
    if np.std(d1) < 1e-6 or np.std(d2) < 1e-6:
        return 0.0
    return float(pearsonr(d1, d2).statistic)
