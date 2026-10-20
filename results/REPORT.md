## Training ablations

| Experiment | Model | imgsz | Epochs | Batch | mAP@50 | mAP@50-95 | Precision | Recall | Params (M) | Train (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| s_640 | yolo11s.pt | 640 | 50 | 8 | 0.386 | 0.224 | 0.514 | 0.399 | 9.4 | 71.8 |

## Tiled inference

| Tag | Split | Mode | Overlap | Merge | Images | mAP@50 | mAP@50-95 | Recall small | Recall medium | Recall large | s / image |
|---|---|---|---|---|---|---|---|---|---|---|---|
| overlap_0.15 | val | tiled | 0.15 | iou@0.6 | 548 | 0.405 | 0.244 | 0.550 | 0.806 | 0.869 | 0.053 |
| overlap_0.15 | val | tiled+full | 0.15 | iou@0.6 | 548 | 0.417 | 0.257 | 0.577 | 0.846 | 0.919 | 0.071 |
| overlap_0.40 | val | tiled | 0.40 | iou@0.6 | 548 | 0.406 | 0.245 | 0.553 | 0.809 | 0.864 | 0.057 |
| overlap_0.40 | val | tiled+full | 0.40 | iou@0.6 | 548 | 0.418 | 0.258 | 0.579 | 0.848 | 0.918 | 0.075 |
| ios_merge | val | tiled | 0.25 | ios@0.6 | 548 | 0.435 | 0.254 | 0.539 | 0.795 | 0.832 | 0.046 |
| ios_merge | val | tiled+full | 0.25 | ios@0.6 | 548 | 0.451 | 0.267 | 0.562 | 0.834 | 0.904 | 0.060 |
| iou_merge | val | full | 0.25 | iou@0.6 | 548 | 0.367 | 0.224 | 0.375 | 0.773 | 0.890 | 0.014 |
| iou_merge | val | tiled | 0.25 | iou@0.6 | 548 | 0.404 | 0.244 | 0.551 | 0.806 | 0.866 | 0.078 |
| iou_merge | val | tiled+full | 0.25 | iou@0.6 | 548 | 0.416 | 0.257 | 0.577 | 0.846 | 0.920 | 0.090 |
| ios_0.4 | val | tiled+full | 0.25 | ios@0.4 | 548 | 0.447 | 0.265 | 0.550 | 0.812 | 0.887 | 0.059 |
| ios_0.5 | val | tiled+full | 0.25 | ios@0.5 | 548 | 0.450 | 0.267 | 0.556 | 0.826 | 0.899 | 0.061 |
| ios_0.7 | val | tiled+full | 0.25 | ios@0.7 | 548 | 0.449 | 0.267 | 0.567 | 0.837 | 0.904 | 0.074 |

## Deployment

_no rows yet_
