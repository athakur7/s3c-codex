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


def save_method_panel(path: Path, panels: list[tuple[str, np.ndarray, str | None]]) -> None:
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axs = [axs]
    for i, (title, img, cmap) in enumerate(panels):
        axs[i].imshow(img, cmap=cmap)
        axs[i].set_title(title)
        axs[i].axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_global_grid_dynamic(
    path: Path,
    rows: list[tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None]],
    method_order: list[str],
    method_titles: dict[str, str],
    rows_per_file: int = 80,
) -> None:
    if rows_per_file <= 0:
        raise ValueError("rows_per_file must be positive.")

    chunks = [rows[i : i + rows_per_file] for i in range(0, len(rows), rows_per_file)]
    for page_index, chunk in enumerate(chunks, start=1):
        page_path = path if page_index == 1 else path.with_name(f"{path.stem}_{page_index:03d}{path.suffix}")
        _save_global_grid_page(
            page_path,
            chunk,
            method_order,
            method_titles,
            page_label=f"{page_index}/{len(chunks)}" if len(chunks) > 1 else None,
        )


def _save_global_grid_page(
    path: Path,
    rows: list[tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None]],
    method_order: list[str],
    method_titles: dict[str, str],
    page_label: str | None,
) -> None:
    cols = 2 + len(method_order)  # original + methods + mask
    n = len(rows)
    fig, axs = plt.subplots(n, cols, figsize=(3 * cols, 2.4 * n))
    if n == 1:
        axs = np.array([axs])
    if page_label is not None:
        fig.suptitle(f"Grid comparison page {page_label}", fontsize=12)
    for i, (orig, method_imgs, mask) in enumerate(rows):
        axs[i, 0].imshow(orig)
        axs[i, 0].set_title("Original")
        axs[i, 0].axis("off")
        for j, method in enumerate(method_order, start=1):
            img = method_imgs.get(method, np.zeros_like(orig))
            axs[i, j].imshow(img)
            axs[i, j].set_title(method_titles.get(method, method))
            axs[i, j].axis("off")
        axs[i, cols - 1].imshow(mask if mask is not None else np.zeros(orig.shape[:2], dtype=np.uint8), cmap="gray")
        axs[i, cols - 1].set_title("GT Mask")
        axs[i, cols - 1].axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=100, pil_kwargs={"optimize": True})
    plt.close(fig)
