from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def save_comparison_panel(
    path: Path,
    original: np.ndarray,
    baseline: np.ndarray | None,
    s3c: np.ndarray | None,
    mask: np.ndarray,
    baseline_text: str,
    s3c_text: str,
    mask_title: str,
) -> None:
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(original)
    axs[0].set_title("Original")
    axs[1].imshow(baseline if baseline is not None else np.zeros_like(original))
    axs[1].set_title(baseline_text)
    axs[2].imshow(s3c if s3c is not None else np.zeros_like(original))
    axs[2].set_title(s3c_text)
    axs[3].imshow(mask, cmap="gray")
    axs[3].set_title(mask_title)
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_global_grid(path: Path, rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> None:
    n = len(rows)
    fig, axs = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axs = np.array([axs])
    for i, (orig, base, s3c, mask) in enumerate(rows):
        axs[i, 0].imshow(orig)
        axs[i, 1].imshow(base)
        axs[i, 2].imshow(s3c)
        axs[i, 3].imshow(mask, cmap="gray")
        for j in range(4):
            axs[i, j].axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
