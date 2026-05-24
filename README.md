# S3C: Shadow- and Structure-Preserving Seam Carving

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Example Run

This command works on both local and Kaggle when your dataset follows one of the supported defaults:

```bash
python run.py \
  --num-images 5 \
  --seed 42 \
  --reduction 0.25 \
  --shadow-mode gt \
  --methods s3c,baseline \
  --alpha 1.0 --beta 2.0 --gamma 2.5 --delta 2.0 \
  --p-shadow 10 --p-struct 10 --tau-struct 0.5 \
  --lambda-sh 1.0 \
  --workers 4 \
  --verbose
```

Explicit local path example:

```bash
python run.py --data-dir ../data/SBU-shadow/SBU-Test --output-root ./outputs --num-images 5
```

Explicit Kaggle path example:

```bash
python run.py --data-dir /kaggle/input/datasets/anandthakur178/sbu-shadow-zip/SBU-shadow/SBU-Test --output-root /kaggle/working/outputs --num-images 5
```

## Kaggle Run (CPU-optimized)

```bash
python run.py \
  --data-dir /kaggle/input/datasets/anandthakur178/sbu-shadow-zip/SBU-shadow/SBU-Test \
  --images-subdir ShadowImages \
  --masks-subdir ShadowMasks \
  --num-images 10 \
  --seed 42 \
  --reduction 0.25 \
  --shadow-mode gt \
  --methods s3c,baseline \
  --executor process \
  --workers 7 \
  --opencv-threads-per-worker 1 \
  --output-root /kaggle/working/outputs
```

Notes:
- `--executor process` is best for CPU-bound seam carving.
- Keep `--opencv-threads-per-worker 1` to avoid CPU oversubscription when using multiple processes.
- By default, `run.py` now uses Kaggle-friendly paths and process execution.

## Expected Output Tree

```text
outputs/
└── run_YYYY-MM-DD_HH-MM-SS/
    ├── config.json
    ├── sampled_images.txt
    ├── metrics.csv
    ├── summary.json
    ├── errors.log
    ├── grid_comparison.png
    └── per_image/
        └── <stem>/
            ├── original.png
            ├── gt_mask.png
            ├── importance_s3c.png
            ├── importance_baseline.png
            ├── seams_s3c.png
            ├── seams_baseline.png
            ├── resized_s3c.png
            ├── resized_baseline.png
            ├── carved_mask_s3c.png
            ├── carved_mask_baseline.png
            └── comparison.png
```

## Map Definitions

- `E_g`: Sobel gradient magnitude on Lab L channel.
- `E_s`: cluster saliency from Lab+Gabor with contrast and center-bias cue.
- `E_sh`: shadow map from GT mask (`gt` mode) or classical Lab local-illumination detector (`auto` mode).
- `E_str`: structure tensor coherence-strength map with line-detector enhancement.

## How is S3C different from Hashemzadeh 2019?

### 2.1 Structure map `E_str` (new fourth map)

Compute a **structure tensor**-based map:

```text
J_σ = G_σ * [[Ix·Ix, Ix·Iy], [Ix·Iy, Iy·Iy]]
λ1, λ2 = eigenvalues of J_σ at each pixel  (λ1 ≥ λ2)
coherence    c = ((λ1 - λ2) / (λ1 + λ2 + ε))^2     # high on lines/edges, low on corners and flats
edge_strength e = λ1 + λ2                          # gradient energy
E_str = normalize(coherence × edge_strength)
```

Augment with a **line-segment detector** (`cv2.createLineSegmentDetector` is deprecated; use `cv2.ximgproc.createFastLineDetector` if `opencv-contrib-python` is installed, otherwise fall back to a Hough-based line mask). Add detected line pixels to `E_str` with weight `w_line` (default 1.5).

`σ` for the Gaussian in the tensor: default 1.5 px, configurable.

### 2.2 Shadow map `E_sh` — two modes

- **`--shadow-mode gt`**: load the corresponding mask from the dataset (mask folder), binarize, light Gaussian feather (σ=1) to soften the boundary. Use this for the SBU images.
- **`--shadow-mode auto`**: improved unsupervised detector. Convert to **CIE Lab**; combine low-L with low chroma divergence from the local illumination model. Specifically:
  - `L_norm = L / 100`
  - illumination invariant ratio: `r = (L_norm + ε) / (mean_L_local + ε)` over a 51×51 window
  - shadow candidate if `r < τ_r` (default 0.65) AND `chroma_diff < τ_c` from local mean
  - morphological open/close to clean, then connected-component filter to drop tiny blobs (< 0.1% image area)
  - This is strictly better than the YCbCr threshold approach because it is illumination-aware and not fooled by globally dark images.

Output `E_sh` ∈ [0, 1] (soft, not binary) so multiplicative fusion is not brittle.

### 2.3 Dual penalty-augmented seam cost (the core algorithmic contribution)

Replace the baseline's pure multiplicative fusion with an **additive-with-penalty** formulation:

```text
E_base(i, j)    = α·E_g  +  β·E_s  +  γ·E_sh  +  δ·E_str
penalty_shadow  = P_sh · 1[boundary_of(E_sh) at (i, j)]
penalty_struct  = P_st · 1[E_str(i, j) > τ_str]
E(i, j)         = E_base(i, j) + penalty_shadow + penalty_struct
```

Defaults: `α=1.0, β=2.0, γ=2.5, δ=2.0`, `P_sh = 10·max(E_base)`, `P_st = 10·max(E_base)`, `τ_str = 0.5`.

### 2.4 Shadow-aware forward energy

Extend Rubinstein's forward-energy cost with an additional shadow-discontinuity term:

```text
C_*_S(i, j) = ‖E_sh(neighbor_after_removal) - E_sh(other_neighbor_after_removal)‖
```

Add `λ_sh · C_*_S` to each forward cost branch.

### 2.5 Seam-consistent mask tracking

Every seam removed from the image is also removed from shadow/auxiliary maps. Metrics include:

- `Shadow Preservation Ratio (SPR)`
- `Shadow IoU`

### 2.6 Anti-aliasing — bilateral averaging

S3C uses a 1D bilateral filter on seam lines (window=5, σ_spatial=1.5, σ_range=15), while baseline uses 2-point averaging.
