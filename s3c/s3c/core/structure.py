from __future__ import annotations

import cv2
import numpy as np

from s3c.utils import normalize_map


def _line_mask(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.float32)
    try:
        fld = cv2.ximgproc.createFastLineDetector()  # type: ignore[attr-defined]
        lines = fld.detect(gray)
        if lines is not None:
            for line in lines.reshape(-1, 4):
                x1, y1, x2, y2 = map(int, line)
                cv2.line(mask, (x1, y1), (x2, y2), 1.0, 1)
            return mask
    except Exception:
        pass

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=max(15, w // 20), maxLineGap=5)
    if lines is not None:
        for l in lines.reshape(-1, 4):
            x1, y1, x2, y2 = map(int, l)
            cv2.line(mask, (x1, y1), (x2, y2), 1.0, 1)
    return mask


def structure_map(image_rgb: np.ndarray, sigma: float = 1.5, w_line: float = 1.5) -> np.ndarray:
    """Compute E_str with structure tensor coherence and line enhancement."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    j11 = cv2.GaussianBlur(ix * ix, (0, 0), sigmaX=sigma, sigmaY=sigma)
    j22 = cv2.GaussianBlur(iy * iy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    j12 = cv2.GaussianBlur(ix * iy, (0, 0), sigmaX=sigma, sigmaY=sigma)

    trace = j11 + j22
    det = j11 * j22 - j12 * j12
    tmp = np.sqrt(np.clip(trace * trace - 4.0 * det, 0.0, None))
    l1 = 0.5 * (trace + tmp)
    l2 = 0.5 * (trace - tmp)
    coherence = ((l1 - l2) / (l1 + l2 + 1e-8)) ** 2
    edge_strength = l1 + l2

    estr = normalize_map(coherence * edge_strength)
    line = _line_mask(image_rgb)
    estr = normalize_map(estr + w_line * line)
    return estr
