from __future__ import annotations

import numpy as np


def cumulative_forward_cost(energy: np.ndarray, c_l: np.ndarray, c_u: np.ndarray, c_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """DP cumulative map + backtrack pointers for vertical seam."""
    h, w = energy.shape
    m = np.zeros((h, w), dtype=np.float32)
    bt = np.zeros((h, w), dtype=np.int8)
    m[0] = energy[0]
    inf = np.float32(1e12)
    for i in range(1, h):
        prev = m[i - 1]
        left_prev = np.empty_like(prev)
        right_prev = np.empty_like(prev)
        left_prev[0] = inf
        left_prev[1:] = prev[:-1]
        right_prev[-1] = inf
        right_prev[:-1] = prev[1:]

        l_cost = left_prev + c_l[i]
        u_cost = prev + c_u[i]
        r_cost = right_prev + c_r[i]
        stacked = np.stack([l_cost, u_cost, r_cost], axis=0)
        choice = np.argmin(stacked, axis=0).astype(np.int8)
        bt[i] = choice - 1
        min_cost = stacked[choice, np.arange(w)]
        m[i] = energy[i] + min_cost
    return m, bt


def backtrack_min_seam(cumulative: np.ndarray, backtrack: np.ndarray) -> np.ndarray:
    h, w = cumulative.shape
    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = int(np.argmin(cumulative[-1]))
    for i in range(h - 2, -1, -1):
        seam[i] = seam[i + 1] + int(backtrack[i + 1, seam[i + 1]])
        seam[i] = max(0, min(w - 1, seam[i]))
    return seam
