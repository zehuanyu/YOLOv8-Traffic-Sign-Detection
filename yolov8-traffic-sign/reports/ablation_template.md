# Ablation Study Template

## Experimental Setup

- Dataset:
- Train/val split:
- Image size:
- Epochs:
- Batch size:
- Device:

## Variants Compared

| Variant | Change | Expected effect |
|---|---|---|
| Baseline | Standard YOLOv8 | Reference point |
| CBAM | Attention module | Improve feature focus |
| LCFE | Lightweight context enhancement | Improve small-object context |
| LCFE v2 | More stable LCFE version | Improve training stability |
| IMCMD | Custom multi-branch feature fusion | Improve representational power |
| IMCMD-TS | IMCMD plus AGRFM | Improve multi-scale fusion |
| TS | YOLO-TS style ablation | Isolate AGRFM contribution |

## Results

| Variant | mAP50 | mAP50-95 | Precision | Recall | Params | Notes |
|---|---|---|---|---|---|---|
| Baseline |  |  |  |  |  |  |
| CBAM |  |  |  |  |  |  |
| LCFE |  |  |  |  |  |  |
| LCFE v2 |  |  |  |  |  |  |
| IMCMD |  |  |  |  |  |  |
| IMCMD-TS |  |  |  |  |  |  |
| TS |  |  |  |  |  |  |

## Takeaways

- Which variant performed best:
- Which variant was most efficient:
- Which modules helped most for small-object detection:
- What should be tried next:
