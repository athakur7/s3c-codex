from __future__ import annotations

import cv2
import numpy as np

from s3c.utils import normalize_map


def _gabor_bank(gray: np.ndarray) -> np.ndarray:
    feats = []
    for theta in np.linspace(0, np.pi, 8, endpoint=False):
        kernel = cv2.getGaborKernel((21, 21), 4.0, float(theta), 10.0, 0.5, 0, ktype=cv2.CV_32F)
        resp = cv2.filter2D(gray, cv2.CV_32F, kernel)
        feats.append(resp)
    return np.stack(feats, axis=-1)


def cluster_saliency_map(image_rgb: np.ndarray, k: int = 6) -> np.ndarray:
    """Cluster-based saliency map approximating Hashemzadeh 2019."""
    h, w = image_rgb.shape[:2]
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gabor = _gabor_bank(gray)
    feats = np.concatenate([lab, gabor], axis=-1).reshape(-1, 11).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _, labels, centers = cv2.kmeans(feats, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(h, w)

    counts = np.array([(labels == i).sum() for i in range(k)], dtype=np.float32)
    centers = centers.astype(np.float32)

    contrast = np.zeros(k, dtype=np.float32)
    for i in range(k):
        d = np.linalg.norm(centers[i][None, :] - centers, axis=1)
        contrast[i] = float(np.sum((counts / (counts.sum() + 1e-8)) * d))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    sx, sy = w / 4.0, h / 4.0
    spatial = np.exp(-(((xx - cx) ** 2) / (2 * sx * sx + 1e-8) + ((yy - cy) ** 2) / (2 * sy * sy + 1e-8)))

    sal = contrast[labels] * spatial
    return normalize_map(sal)
