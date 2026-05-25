from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# Ensure local package resolution (important in Kaggle where another `s3c` can exist).
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from s3c.baseline.basic import run_basic
from s3c.baseline.hashemzadeh import run_baseline
from s3c.config import S3CConfig
from s3c.data.loader import ImageSample, discover_samples, sample_images
from s3c.metrics.image_metrics import ssim_vs_rescaled
from s3c.metrics.shadow_metrics import shadow_iou, shadow_preservation_ratio
from s3c.metrics.structure_metrics import structure_edge_corr
from s3c.s3c_method import run_s3c
from s3c.utils import configure_logging, normalize_map, read_rgb, save_json, save_png, timestamped_run_dir
from s3c.viz.compare import save_global_grid_dynamic, save_method_panel

logger = logging.getLogger("s3c.run")


def _default_data_dir() -> str:
    candidates = [
        Path("/kaggle/input/datasets/anandthakur178/sbu-shadow-zip/SBU-shadow/SBU-Test"),
        Path("/kaggle/input") / "datasets" / "anandthakur178" / "sbu-shadow-zip" / "SBU-shadow" / "SBU-Test",
        Path(__file__).resolve().parent.parent / "data" / "SBU-shadow" / "SBU-Test",
        Path("./data/SBU-shadow/SBU-Test"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[0])


def _default_output_root() -> str:
    if Path("/kaggle/working").exists():
        return "/kaggle/working/outputs"
    return "./outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        default=_default_data_dir(),
        help="Dataset root. Auto-detects Kaggle or local repo layout.",
    )
    p.add_argument("--images-subdir", default="ShadowImages")
    p.add_argument("--masks-subdir", default="ShadowMasks")
    p.add_argument("--num-images", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--reduction", type=float, default=0.25)
    p.add_argument("--shadow-mode", choices=["gt", "auto"], default="auto")
    p.add_argument("--methods", default="s3c,baseline")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=2.5)
    p.add_argument("--delta", type=float, default=2.0)
    p.add_argument("--p-shadow", type=float, default=10.0)
    p.add_argument("--p-struct", type=float, default=10.0)
    p.add_argument("--tau-struct", type=float, default=0.5)
    p.add_argument("--lambda-sh", type=float, default=1.0)
    p.add_argument("--output-root", default=_default_output_root())
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--executor", choices=["serial", "thread", "process"], default="process")
    p.add_argument("--opencv-threads-per-worker", type=int, default=1)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def to_config(args: argparse.Namespace) -> S3CConfig:
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    if not methods:
        raise ValueError("No methods provided in --methods.")
    if args.num_images != "all":
        int(args.num_images)
    return S3CConfig(
        data_dir=args.data_dir,
        images_subdir=args.images_subdir,
        masks_subdir=args.masks_subdir,
        num_images=str(args.num_images),
        seed=args.seed,
        reduction=args.reduction,
        shadow_mode=args.shadow_mode,
        methods=methods,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        p_shadow=args.p_shadow,
        p_struct=args.p_struct,
        tau_struct=args.tau_struct,
        lambda_sh=args.lambda_sh,
        output_root=args.output_root,
        workers=args.workers,
        executor=args.executor,
        opencv_threads_per_worker=args.opencv_threads_per_worker,
        verbose=args.verbose,
    )


def _load_gt_mask(sample: ImageSample, shape: tuple[int, int]) -> np.ndarray | None:
    if sample.mask_path is None:
        return None
    import cv2

    m = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.float32)


def process_one(sample: ImageSample, config: S3CConfig, run_dir: Path) -> dict[str, Any]:
    import cv2

    cv2.setNumThreads(max(0, config.opencv_threads_per_worker))
    image = read_rgb(sample.image_path)
    h, w = image.shape[:2]
    target_w = max(1, int(round(w * (1.0 - config.reduction))))
    gt_mask = _load_gt_mask(sample, (h, w))
    img_dir = run_dir / "per_image" / sample.image_id
    img_dir.mkdir(parents=True, exist_ok=True)
    save_png(img_dir / "original.png", image)
    if gt_mask is not None:
        save_png(img_dir / "gt_mask.png", gt_mask)

    results: dict[str, Any] = {"image_id": sample.image_id, "orig_h": h, "orig_w": w, "new_w": target_w}
    renders: dict[str, Any] = {}

    if "baseline" in config.methods:
        baseline = run_baseline(image, target_w, config, initial_mask=gt_mask)
        save_png(img_dir / "importance_baseline.png", normalize_map(baseline.importance))
        save_png(img_dir / "seams_baseline.png", baseline.seam_overlay)
        save_png(img_dir / "resized_baseline.png", baseline.image)
        save_png(img_dir / "carved_mask_baseline.png", baseline.mask)
        renders["baseline"] = baseline.image
        results["baseline"] = {
            "runtime_sec": baseline.runtime_sec,
            "shadow_iou": shadow_iou(gt_mask, baseline.mask, baseline.mask.shape),
            "shadow_preservation_ratio": shadow_preservation_ratio(gt_mask, baseline.mask, target_w / w),
            "structure_edge_corr": structure_edge_corr(image, baseline.image),
            "ssim_vs_rescaled": ssim_vs_rescaled(image, baseline.image),
            "seam_through_shadow_pct": baseline.seam_through_shadow_pct,
        }

    if "basic" in config.methods:
        basic = run_basic(image, target_w, initial_mask=gt_mask)
        save_png(img_dir / "importance_basic.png", normalize_map(basic.importance))
        save_png(img_dir / "seams_basic.png", basic.seam_overlay)
        save_png(img_dir / "resized_basic.png", basic.image)
        save_png(img_dir / "carved_mask_basic.png", basic.mask)
        renders["basic"] = basic.image
        results["basic"] = {
            "runtime_sec": basic.runtime_sec,
            "shadow_iou": shadow_iou(gt_mask, basic.mask, basic.mask.shape),
            "shadow_preservation_ratio": shadow_preservation_ratio(gt_mask, basic.mask, target_w / w),
            "structure_edge_corr": structure_edge_corr(image, basic.image),
            "ssim_vs_rescaled": ssim_vs_rescaled(image, basic.image),
            "seam_through_shadow_pct": basic.seam_through_shadow_pct,
        }

    if "s3c" in config.methods:
        s3c_res = run_s3c(image, target_w, config, sample.mask_path, initial_mask=gt_mask)
        save_png(img_dir / "importance_s3c.png", normalize_map(s3c_res.importance))
        save_png(img_dir / "seams_s3c.png", s3c_res.seam_overlay)
        save_png(img_dir / "resized_s3c.png", s3c_res.image)
        save_png(img_dir / "carved_mask_s3c.png", s3c_res.mask)
        renders["s3c"] = s3c_res.image
        results["s3c"] = {
            "runtime_sec": s3c_res.runtime_sec,
            "shadow_iou": shadow_iou(gt_mask, s3c_res.mask, s3c_res.mask.shape),
            "shadow_preservation_ratio": shadow_preservation_ratio(gt_mask, s3c_res.mask, target_w / w),
            "structure_edge_corr": structure_edge_corr(image, s3c_res.image),
            "ssim_vs_rescaled": ssim_vs_rescaled(image, s3c_res.image),
            "seam_through_shadow_pct": s3c_res.seam_through_shadow_pct,
            "used_shadow_mode": s3c_res.used_shadow_mode,
        }

    base_img = renders.get("baseline", np.zeros_like(image))
    s3c_img = renders.get("s3c", np.zeros_like(image))
    if gt_mask is not None:
        mask_vis = gt_mask
    elif "s3c" in config.methods:
        mask_vis = (s3c_res.mask > 0.5).astype(np.float32)  # noqa: F821
    elif "baseline" in config.methods:
        mask_vis = (baseline.mask > 0.5).astype(np.float32)  # noqa: F821
    else:
        mask_vis = np.zeros((h, target_w), dtype=np.float32)
    method_labels = {"basic": "Basic", "baseline": "Hashemzadeh", "s3c": "S3C"}
    dynamic_panels: list[tuple[str, np.ndarray, str | None]] = [("Original", image, None)]
    for method_name in [m for m in ("basic", "baseline", "s3c") if m in config.methods]:
        if method_name not in renders:
            continue
        label = method_labels[method_name]
        met = results.get(method_name, {})
        if "shadow_preservation_ratio" in met and "shadow_iou" in met:
            label = f"{label} - SPR {met['shadow_preservation_ratio']:.2f}, IoU {met['shadow_iou']:.2f}"
        dynamic_panels.append((label, renders[method_name], None))
    if gt_mask is not None:
        dynamic_panels.append(("GT Mask", gt_mask, "gray"))
    else:
        dynamic_panels.append(("Shadow Map Used", mask_vis, "gray"))
    save_method_panel(img_dir / "comparison.png", dynamic_panels)
    save_method_panel(img_dir / "comparison_methods.png", dynamic_panels)
    return results


def main() -> None:
    args = parse_args()
    cfg = to_config(args)
    root = Path(cfg.output_root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = timestamped_run_dir(root)
    configure_logging(run_dir / "errors.log", verbose=cfg.verbose)
    save_json(run_dir / "config.json", cfg)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    samples_all = discover_samples(Path(cfg.data_dir), cfg.images_subdir, cfg.masks_subdir)
    samples = sample_images(samples_all, cfg.num_images, cfg.seed)
    names = [s.image_path.name for s in samples]
    print("Sampled filenames:")
    for n in names:
        print(f" - {n}")
    (run_dir / "sampled_images.txt").write_text("\n".join(names), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    grid_rows: list[tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None]] = []
    success = 0
    def _collect_result(res: dict[str, Any]) -> None:
        nonlocal success
        success += 1
        image_id = res["image_id"]
        orig_h, orig_w, new_w = res["orig_h"], res["orig_w"], res["new_w"]
        for method in cfg.methods:
            if method not in res:
                continue
            met = dict(res[method])
            rows.append({"image_id": image_id, "method": method, "orig_h": orig_h, "orig_w": orig_w, "new_w": new_w, **met})
        img_dir = run_dir / "per_image" / image_id
        orig = read_rgb(img_dir / "original.png")
        method_imgs: dict[str, np.ndarray] = {}
        for method in ("basic", "baseline", "s3c"):
            p = img_dir / f"resized_{method}.png"
            if p.exists():
                method_imgs[method] = read_rgb(p)
        mask = np.array(read_rgb(img_dir / "gt_mask.png"))[:, :, 0] if (img_dir / "gt_mask.png").exists() else None
        grid_rows.append((orig, method_imgs, mask))

    if cfg.executor == "serial" or cfg.workers <= 1:
        for sample in tqdm(samples, total=len(samples), desc="Processing images"):
            try:
                _collect_result(process_one(sample, cfg, run_dir))
            except Exception as e:
                logger.exception("Failed image %s: %s", sample.image_id, e)
    elif cfg.executor == "thread":
        with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
            futs = {ex.submit(process_one, s, cfg, run_dir): s for s in samples}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Processing images"):
                sample = futs[fut]
                try:
                    _collect_result(fut.result())
                except Exception as e:
                    logger.exception("Failed image %s: %s", sample.image_id, e)
    else:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            futs = {ex.submit(process_one, s, cfg, run_dir): s for s in samples}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Processing images"):
                sample = futs[fut]
                try:
                    _collect_result(fut.result())
                except Exception as e:
                    logger.exception("Failed image %s: %s", sample.image_id, e)

    if grid_rows:
        method_order = [m for m in ("basic", "baseline", "s3c") if m in cfg.methods]
        method_titles = {"basic": "Basic", "baseline": "Hashemzadeh", "s3c": "S3C"}
        save_global_grid_dynamic(run_dir / "grid_comparison.png", grid_rows, method_order, method_titles)

    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        if rows:
            keys = sorted({k for r in rows for k in r.keys()})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    summary: dict[str, Any] = {"success_rate": success / max(1, len(samples)), "n_samples": len(samples), "n_success": success}
    for method in cfg.methods:
        mrows = [r for r in rows if r["method"] == method]
        if not mrows:
            continue
        method_summary = {}
        for col in ["runtime_sec", "shadow_iou", "shadow_preservation_ratio", "structure_edge_corr", "ssim_vs_rescaled", "seam_through_shadow_pct"]:
            vals = np.array([r[col] for r in mrows], dtype=np.float32)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            method_summary[col] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
        summary[method] = method_summary

    if "s3c" in summary and "baseline" in summary:
        diag = {}
        for col in ["shadow_iou", "shadow_preservation_ratio"]:
            if col in summary["s3c"] and col in summary["baseline"]:
                diag[col] = summary["s3c"][col]["mean"] > summary["baseline"][col]["mean"]
        if "seam_through_shadow_pct" in summary["s3c"] and "seam_through_shadow_pct" in summary["baseline"]:
            diag["seam_through_shadow_pct_half_or_better"] = (
                summary["s3c"]["seam_through_shadow_pct"]["mean"] <= 0.5 * summary["baseline"]["seam_through_shadow_pct"]["mean"]
            )
        summary["s3c_vs_baseline_diagnostic"] = diag
        if not all(diag.values()):
            logger.warning("S3C did not beat baseline on all required criteria: %s", diag)

    save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
