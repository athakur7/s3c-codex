from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(slots=True)
class ImageSample:
    image_path: Path
    mask_path: Path | None
    image_id: str


def _list_images(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


def _find_mask_for_image(mask_dir: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def discover_samples(data_dir: Path, images_subdir: str, masks_subdir: str) -> list[ImageSample]:
    split_images = data_dir / images_subdir
    split_masks = data_dir / masks_subdir

    if split_images.exists() and split_images.is_dir():
        if not split_masks.exists() or not split_masks.is_dir():
            raise FileNotFoundError(
                f"Found images subdir '{split_images}', but masks subdir '{split_masks}' is missing."
            )
        images = _list_images(split_images)
        if not images:
            raise FileNotFoundError(f"No images found in '{split_images}'.")
        return [
            ImageSample(
                image_path=img,
                mask_path=_find_mask_for_image(split_masks, img.stem),
                image_id=img.stem,
            )
            for img in images
        ]

    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    images = _list_images(data_dir)
    if not images:
        raise FileNotFoundError(
            f"No images found in '{data_dir}'. Expected either a flat image folder or subfolders "
            f"'{images_subdir}' and '{masks_subdir}'."
        )
    masks = {p.stem: p for p in images if "mask" in p.stem.lower()}
    return [ImageSample(image_path=img, mask_path=masks.get(img.stem), image_id=img.stem) for img in images]


def sample_images(samples: list[ImageSample], num_images: str, seed: int | None) -> list[ImageSample]:
    if num_images == "all":
        return list(samples)
    n = int(num_images)
    if n <= 0:
        raise ValueError("--num-images must be > 0 or 'all'.")
    if n > len(samples):
        raise ValueError(f"Requested {n} images but only found {len(samples)}")
    rng = random.Random(seed)
    return rng.sample(samples, n)
