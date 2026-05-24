# S3C Results (Example Run)

Run folder: `C:\Users\AnandThakur\Desktop\s3c-codex\s3c\outputs\run_2026-05-24_21-34-54`

## Configuration

- Dataset: `..\data\SBU-shadow\SBU-Test`
- Images sampled: 3
- Seed: 42
- Reduction: 0.03
- Methods: `s3c,baseline`

## Aggregate Metrics (mean ± std)

| Metric | Baseline | S3C |
|---|---:|---:|
| runtime_sec | 3.7568 ± 1.6959 | 5.5211 ± 2.7866 |
| shadow_iou | 0.9180 ± 0.0421 | **0.9180 ± 0.0421** |
| shadow_preservation_ratio | 0.9732 ± 0.0442 | **0.9732 ± 0.0442** |
| structure_edge_corr | 0.9156 ± 0.0535 | **0.9160 ± 0.0533** |
| ssim_vs_rescaled | 0.6165 ± 0.0795 | **0.6166 ± 0.0796** |
| seam_through_shadow_pct (lower better) | 30.0702 ± 15.8105 | 30.0638 ± 15.8055 |

Diagnostic from `summary.json`:

- `shadow_iou`: pass
- `shadow_preservation_ratio`: pass
- `seam_through_shadow_pct_half_or_better`: fail

## Visual Comparisons

![comparison 1](C:/Users/AnandThakur/Desktop/s3c-codex/s3c/outputs/run_2026-05-24_21-34-54/per_image/80251-732c0-47245934-m750x740-u7c59c/comparison.png)

![comparison 2](C:/Users/AnandThakur/Desktop/s3c-codex/s3c/outputs/run_2026-05-24_21-34-54/per_image/1310391610/comparison.png)

![comparison 3](C:/Users/AnandThakur/Desktop/s3c-codex/s3c/outputs/run_2026-05-24_21-34-54/per_image/DSCN0531/comparison.png)
