# S3C: Shadow- and Structure-Preserving Seam Carving

This repository implements three content-aware width-reduction methods for head-to-head comparison:

- `basic`: classic seam carving (backward energy)
- `baseline`: Hashemzadeh-style shadow-preserving seam carving (2019-inspired)
- `s3c`: proposed Shadow- and Structure-Preserving Seam Carving (S3C)

It produces per-image visual artifacts plus aggregate metrics to verify whether S3C outperforms baseline methods.

## What We Built

- Full CPU pipeline in Python (no deep learning):
  - Gradient, saliency, shadow, and structure maps
  - Forward-energy seam DP
  - Seam-consistent map/mask carving
  - Per-image and batch-level visualization
  - CSV/JSON metrics output
- Optional 2-method or 3-method comparison in one command:
  - `--methods s3c,baseline`
  - `--methods basic,baseline,s3c`
- Local and Kaggle-compatible runner with process-level parallelism.

## Repository Layout

```text
.
├── run.py
├── requirements.txt
├── README.md
├── RESULTS.md
├── s3c/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── s3c_method.py
│   ├── core/
│   ├── carve/
│   ├── baseline/
│   ├── metrics/
│   ├── data/
│   └── viz/
└── tests/
```

## Installation

## Local (Windows/Linux/macOS)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
pip install -r requirements.txt
```

## Kaggle

```bash
!git clone https://github.com/athakur7/s3c-codex.git /kaggle/working/s3c-codex
%cd /kaggle/working/s3c-codex
!pip install -r requirements.txt -q
```

## Dataset Expectations

Expected split layout:

```text
<data-dir>/
├── ShadowImages/
└── ShadowMasks/
```

Image and mask are paired by filename stem, for example:

- `ShadowImages/lssd1.jpg`
- `ShadowMasks/lssd1.png`

If `--shadow-mode gt` is selected and a mask is missing for an image, that image falls back to `auto` shadow mode with a warning.

## How to Run

## Common options

- `--num-images N` or `--num-images all`
- `--reduction 0.25` (remove 25% of width)
- `--methods basic,baseline,s3c` (or any subset)
- `--executor process --workers K --opencv-threads-per-worker 1`
- `--output-root <folder>`

## Local examples

### 2-method (S3C vs baseline)

```bash
python run.py \
  --data-dir ./data/SBU-shadow/SBU-Test \
  --images-subdir ShadowImages \
  --masks-subdir ShadowMasks \
  --num-images 100 \
  --seed 42 \
  --reduction 0.25 \
  --shadow-mode gt \
  --methods s3c,baseline \
  --executor process \
  --workers 3 \
  --opencv-threads-per-worker 1 \
  --output-root ./outputs \
  --verbose
```

### 3-method (Basic + baseline + S3C)

```bash
python run.py \
  --data-dir ./data/SBU-shadow/SBU-Test \
  --images-subdir ShadowImages \
  --masks-subdir ShadowMasks \
  --num-images 300 \
  --seed 42 \
  --reduction 0.25 \
  --shadow-mode gt \
  --methods basic,baseline,s3c \
  --alpha 1.0 --beta 2.0 --gamma 3.0 --delta 3.0 \
  --p-shadow 30 --p-struct 20 --tau-struct 0.35 --lambda-sh 3.0 \
  --executor process \
  --workers 3 \
  --opencv-threads-per-worker 1 \
  --output-root ./outputs
```

## Kaggle example

```bash
!python run.py \
  --data-dir /kaggle/input/datasets/anandthakur178/sbu-shadow-zip/SBU-shadow/SBU-Test \
  --images-subdir ShadowImages \
  --masks-subdir ShadowMasks \
  --num-images 100 \
  --seed 42 \
  --reduction 0.25 \
  --shadow-mode gt \
  --methods basic,baseline,s3c \
  --executor process \
  --workers 3 \
  --opencv-threads-per-worker 1 \
  --output-root /kaggle/working/outputs \
  --verbose
```

## Output Artifacts

Each run creates a fresh timestamped folder:

```text
outputs/
└── run_YYYY-MM-DD_HH-MM-SS/
    ├── config.json
    ├── sampled_images.txt
    ├── metrics.csv
    ├── summary.json
    ├── errors.log
    ├── grid_comparison.png
    └── per_image/<image_id>/
        ├── original.png
        ├── gt_mask.png
        ├── importance_basic.png         # if basic selected
        ├── importance_baseline.png      # if baseline selected
        ├── importance_s3c.png           # if s3c selected
        ├── seams_basic.png
        ├── seams_baseline.png
        ├── seams_s3c.png
        ├── resized_basic.png
        ├── resized_baseline.png
        ├── resized_s3c.png
        ├── carved_mask_basic.png
        ├── carved_mask_baseline.png
        ├── carved_mask_s3c.png
        ├── comparison.png               # dynamic method panel with metrics
        └── comparison_methods.png       # same dynamic method panel
```

## Metrics and Meaning

Logged in `metrics.csv` per `(image_id, method)`:

- `runtime_sec`: per-image wall time
- `shadow_iou`: IoU between resized original GT mask and carved mask
- `shadow_preservation_ratio`: carved shadow area / expected scaled area
- `structure_edge_corr`: Pearson correlation of edge-density grids
- `ssim_vs_rescaled`: SSIM against naively resized original
- `seam_through_shadow_pct`: percent of removed seam pixels crossing shadow

`summary.json` stores means/std and includes:

- `s3c_vs_baseline_diagnostic.shadow_iou`
- `s3c_vs_baseline_diagnostic.shadow_preservation_ratio`
- `s3c_vs_baseline_diagnostic.seam_through_shadow_pct_half_or_better`

## How S3C Differs from Baseline

S3C extends the baseline with:

1. Structure map (`E_str`) from structure tensor coherence and line cues.
2. Dual shadow-map modes:
   - GT-feathered map (`gt`)
   - Lab-based classical detector (`auto`)
3. Additive dual-penalty energy:
   - shadow-boundary crossing penalty
   - high-structure crossing penalty
4. Shadow-aware forward energy branch term.
5. Seam-consistent mask/map tracking.
6. Bilateral seam smoothing (instead of simple 2-point averaging).

## Reproducibility

- Use `--seed` for deterministic sample selection.
- `sampled_images.txt` records exact filenames.
- `config.json` records full hyperparameters per run.

## Testing

Run smoke test:

```bash
python -m pytest -q
```

## Notes and Current Status

- The pipeline is stable on local and Kaggle.
- S3C consistently improves shadow IoU and shadow preservation ratio over baseline in tested runs.
- The strict criterion `seam_through_shadow_pct <= 50% of baseline` is improved substantially but may still fail on some batches; this is explicitly logged in `summary.json` and `errors.log`.
