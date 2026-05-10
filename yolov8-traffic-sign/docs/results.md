# Results Summary

This document summarizes the main experiment results presented in the repository figures and result tables.

## Overall Ranking

| Rank | Model | mAP@0.5 | Parameters |
|---|---|---:|---:|
| 1 | IMCMD | 42.74 | 1.98M |
| 2 | IMCMD-TS | 40.57 | 2.21M |
| 3 | YOLOv8s Baseline | 39.14 | 11.14M |
| 4 | YOLO-TS | 38.10 | 13.71M |
| 5 | CCA_Light | 29.91 | - |
| 6 | CBAM | 11.73 | - |

## Day vs Night Performance

| Model | Day mAP@0.5 | Night mAP@0.5 | Gap | Relative drop |
|---|---:|---:|---:|---:|
| Baseline | 45.62 | 39.09 | 6.53 | 14.3% |
| IMCMD | 45.35 | 40.09 | 5.26 | 11.6% |
| IMCMD-TS | 44.33 | 39.47 | 4.86 | 11.0% |
| YOLO-TS | 42.58 | 36.28 | 6.30 | 14.8% |

## Key Takeaways

- IMCMD achieves the best overall mAP@0.5 while remaining highly parameter-efficient.
- IMCMD-TS is competitive and shows the smallest relative day-to-night drop.
- The custom variants outperform the standard YOLOv8 baseline in the reported setup.
- The ablation comparison suggests that the custom feature-fusion design, especially the C2f_CA based branch, is a major contributor to the performance gain.

## Related Files

- [../charts/chart1_performance_comparison.png](../charts/chart1_performance_comparison.png)
- [../charts/chart2_day_night_comparison.png](../charts/chart2_day_night_comparison.png)
- [../charts/chart3_ablation_study.png](../charts/chart3_ablation_study.png)
- [../charts/chart4_training_convergence.png](../charts/chart4_training_convergence.png)
- [../reports/metrics_summary.csv](../reports/metrics_summary.csv)
- [../reports/day_night_summary.csv](../reports/day_night_summary.csv)
