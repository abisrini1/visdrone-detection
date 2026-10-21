# Small-Object Detection on Drone Imagery (VisDrone)

Detecting pedestrians and vehicles in drone images, where objects are tiny and crowded, and taking the detector from training to a measured, optimized deployment.

The goal is not just a high mAP number. Each step answers a question with evidence:

1. How well does a fine-tuned YOLO do, and *where* does it fail? (baseline + per-class + by object size)
2. What actually helps small objects: resolution, model size, augmentation, or tiled inference?
3. How much speed can be gained from ONNX / TensorRT, and at what accuracy cost?
4. Can detections be tracked stably across frames (ByteTrack)?

## Key findings (VisDrone val, single seed)

These are validation-set results. The single held-out test-dev evaluation is run last, after settings are fixed (see step 7 below).

- **Training resolution is the biggest lever.** YOLO11s goes from 0.386 to 0.488 to 0.538 mAP@50 at 640, 960 and 1280 px, with the largest gains on the smallest classes (bicycle 0.131 to 0.269 to 0.339). Returns diminish: the second step costs about 1.8x the training time for half the gain.
- **Resolution beat model size at equal training cost.** YOLO11m at 640 px (20.1M parameters) and YOLO11s at 960 px (9.4M parameters) took about the same time to train (155 vs 154 min), and the smaller model at higher resolution won: 0.488 vs 0.451 mAP@50 and 0.292 vs 0.269 mAP@50-95. Inference cost of the two is compared in the deployment table.
- **Tiled inference helps on top of resolution, but the gain shrinks as resolution grows.** Tiling adds +8.5 / +7.1 / +4.4 mAP@50 at 640 / 960 / 1280 (own evaluator), at several times the inference cost.
- **A single-pass 960 model beats a tiled 640 model on strict accuracy** (mAP@50-95 0.288 vs 0.267) at a fraction of the cost.
- **Merging with intersection-over-smaller-area (IoS) beats IoU** for combining tile detections (0.451 vs 0.416 mAP@50 for the 640 model), probably because objects cut by a tile border produce partial boxes that IoU does not merge. This explanation is a hypothesis, not tested.
- **Augmentation changes were small.** Stronger scale jitter did not help (0.380 vs 0.386 mAP@50), and turning mosaic off cost about a point (0.374). Both are close enough to single-seed noise that I would not claim an effect.

## Results

### Training ablations (VisDrone val, 50 epochs unless noted)

Ultralytics' validator, conf 0.001, `best.pt`.

| Experiment | Model | imgsz | mAP@50 | mAP@50-95 | Train time | Takeaway |
|---|---|---|---|---|---|---|
| s_640 (baseline) | YOLO11s | 640 | 0.386 | 0.224 | 72 min | Weakest on bicycle, awning-tricycle, people |
| s_960 | YOLO11s | 960 | 0.488 | 0.292 | 154 min | +10.2 mAP@50; small classes gain most |
| s_1280 | YOLO11s | 1280 | 0.538 | 0.332 | 280 min | +5.1 more; diminishing returns, near native image size |
| n_640 | YOLO11n | 640 | 0.307 | 0.170 | 56 min | 2.6M params; -7.9 mAP@50 vs s |
| m_640 | YOLO11m | 640 | 0.451 | 0.269 | 155 min | 20.1M params; +6.5 mAP@50 vs s, but less than raising resolution |
| s_640_no_mosaic | YOLO11s | 640 | 0.374 | 0.217 | 69 min | Slightly worse (-1.2); at the edge of noise |
| s_640_scale_0.9 | YOLO11s | 640 | 0.380 | 0.219 | 71 min | No clear effect (within noise) |

Auto-filled in `results/results.csv`; per-class numbers in `results/per_class/`.

### Tiled inference (own implementation, same evaluator for every row)

Evaluated on the full val set (548 images) with my own COCO-style evaluator. Tiles are 640 px with 25% overlap unless noted; "tiled + full" also runs the whole image and merges everything. The absolute numbers sit about 2 mAP@50 points below Ultralytics' validator (see limitations), so only compare rows within this table.

| Model | Mode | Merge | mAP@50 | mAP@50-95 | Recall small |
|---|---|---|---|---|---|
| s_640 | full image | n/a | 0.367 | 0.224 | 0.375 |
| s_640 | tiled | IoU 0.6 | 0.404 | 0.244 | 0.551 |
| s_640 | tiled + full | IoU 0.6 | 0.416 | 0.257 | 0.577 |
| s_640 | tiled + full | IoS 0.6 | 0.451 | 0.267 | 0.562 |
| s_960 | full image | n/a | 0.460 | 0.288 | 0.498 |
| s_960 | tiled + full | IoS 0.6 | 0.531 | 0.324 | 0.651 |
| s_1280 | full image | n/a | 0.515 | 0.330 | 0.564 |
| s_1280 | tiled + full | IoS 0.6 | 0.559 | 0.347 | 0.672 |
| s_1280 | tiled + full, 1280 px tiles | IoS 0.6 | 0.539 | 0.339 | 0.597 |

Size buckets use sqrt(box area) in pixels: small < 32, medium 32-96, large > 96. Medium and large recall, overlap sweeps (0.15 / 0.40, no meaningful change) and merge-threshold sweeps (IoS 0.4 to 0.7, flat) are in `results/tiled.csv`. Per-image timings from this script are noisy on a laptop and are not reported here; use the benchmark below for latency.

### Deployment (batch 1, hardware: RTX 5060 Laptop GPU, 8 GB)

| Format | Precision | mAP@50-95 | Latency mean (ms) | p95 (ms) | FPS | GPU mem (MiB) |
|---|---|---|---|---|---|---|
| PyTorch | FP32 | TBD | TBD | TBD | TBD | TBD |
| PyTorch | FP16 | TBD | TBD | TBD | TBD | TBD |
| ONNX | FP32 | TBD | TBD | TBD | TBD | TBD |
| TensorRT | FP16 | TBD | TBD | TBD | TBD | TBD |
| TensorRT | INT8 | TBD | TBD | TBD | TBD | TBD |

Auto-filled in `results/benchmark.csv`; run `python -m src.plot_pareto` for the accuracy-vs-latency chart.

### Tracking

| Video | Frames | Proc. FPS | Unique IDs | Mean track length | Tracks < 5 frames |
|---|---|---|---|---|---|
| TBD | | | | | |

### Failure analysis

TBD: add a few annotated images where the model fails (tiny objects, dense crowds, occlusion, motion blur) and what you learned.

## Repo layout

```
src/
  train.py          train one experiment, validate, log a row to results/results.csv
  run_ablations.py  run the experiment queue in configs/experiments.yaml
  evaluate.py       Ultralytics validation + per-class results
  tiling.py         tiled inference + NMS merge (from scratch)
  metrics_np.py     COCO-style mAP + recall by object size (from scratch, unit tested)
  eval_tiled.py     full vs tiled vs tiled+full under one evaluator
  export.py         ONNX / TensorRT (FP16, INT8) export
  benchmark.py      latency / FPS / GPU memory (+ accuracy with --val)
  track.py          detection + ByteTrack on video, with track statistics
  plot_pareto.py    accuracy vs latency plot
  report.py         results/*.csv -> markdown tables (results/REPORT.md)
configs/experiments.yaml   the experiment plan
scripts/check_env.py       verifies PyTorch sees your GPU
tests/                     unit tests for tiling and metrics
results/                   CSV tables, JSON, plots (committed)
```

## Setup

Requires Python 3.9-3.12 and an NVIDIA GPU with a recent driver (CPU works for tests, but training needs a GPU).

```bash
git clone https://github.com/abisrini1/visdrone-detection
cd visdrone-detection
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

**Install PyTorch with CUDA first.** Open https://pytorch.org/get-started/locally/, pick your OS, *Pip* and a CUDA version, and run the command it gives you. Newer GPUs (RTX 50-series) need a recent CUDA build. Then:

```bash
pip install -r requirements.txt
python scripts/check_env.py
```

`check_env.py` should report `CUDA yes` and your GPU. If it says NO, the PyTorch install is CPU-only: redo the previous step.

VisDrone (~2 GB) downloads automatically the first time you train or validate. To store datasets somewhere else, run `yolo settings datasets_dir=/path/to/datasets`.

Run the tests (no GPU or dataset needed):

```bash
python -m pytest -q
```

## Reproduce the experiments

```bash
# 1. Pipeline check (minutes). Confirms training, validation and logging all work.
python -m src.run_ablations --only smoke

# 2. Baseline, then the ablation queue (hours; resume after interruptions with --skip-done)
python -m src.run_ablations --only s_640
python -m src.run_ablations --skip-done

# 3. Tiled inference vs full image
python -m src.eval_tiled --weights runs/s_640/weights/best.pt --imgsz 640 --tile 640 --overlap 0.25
python -m src.eval_tiled --weights runs/s_640/weights/best.pt --imgsz 640 --merge-metric ios --tag ios_merge

# 4. Export, then benchmark every format (with accuracy)
python -m src.export --weights runs/s_640/weights/best.pt --imgsz 640 --formats onnx trt-fp16 trt-int8
python -m src.benchmark --weights runs/s_640/weights/best.pt --imgsz 640 --val
python -m src.benchmark --weights runs/s_640/weights/best.pt --imgsz 640 --val --half
python -m src.benchmark --weights exports/s_640_640_trt-fp16.engine --imgsz 640 --val
# latency of the tiled pipeline at deployment settings (conf 0.25), several repeats
python -m src.benchmark --weights runs/s_640/weights/best.pt --mode tiled+full --merge-metric ios --repeats 3
python -m src.plot_pareto

# 5. Tracking on a drone video
python -m src.track --weights runs/s_640/weights/best.pt --source path/to/drone.mp4 --imgsz 960

# 6. Build the README tables from every results CSV
python -m src.report

# 7. Final numbers: choose settings on val, then evaluate ONCE on the held-out test-dev split
python -m src.eval_tiled --weights runs/s_640/weights/best.pt --split test --tag final
```

Any extra training option can be passed straight through, e.g. `python -m src.train --name my_run --epochs 30 --mosaic 0.0`.

### Hardware notes

Developed on an 8 GB laptop GPU (RTX 5060) with 16 GB RAM using mixed precision. Batch sizes in `configs/experiments.yaml` are sized for 8 GB of VRAM; on a bigger GPU raise them. 1280 px training needs a tiny batch on 8 GB. Ultralytics keeps the effective batch at 64 through gradient accumulation, so per-step batch size differences between experiments do not change the optimization. If data loading hangs (seen on Windows), set `workers: 0`.

VisDrone images contain up to ~900 objects. Ultralytics raises `max_det` automatically for its own training and validation; this repo's evaluation scripts (`eval_tiled`) default to `--max-det 1000` for the same reason (the usual 300 would cap recall in crowded scenes).

TensorRT needs the NVIDIA TensorRT package, which can be fiddly on Windows. If `trt-*` exports fail, ONNX and PyTorch FP16 still give a useful comparison, and WSL2 is an option for TensorRT.

## Honest limitations

- **Single seed.** Every number is from one training run. Gaps of several points (the resolution steps, tiling on the 640 model) are far larger than typical seed variation, but differences of about a point or less, such as the augmentation ablations, should not be read as real effects.
- **My evaluator disagrees with Ultralytics'.** On the same weights my mAP@50 is 1.9 to 2.7 points lower (e.g. 0.515 vs 0.538 for s_1280), while mAP@50-95 agrees to within about 0.5 points. I have not found the cause. Relative comparisons inside my evaluator are consistent, but the absolute values should not be compared with other published numbers.
- **Tuned on val.** Tile size, overlap and merge settings were chosen by looking at val results. The final numbers come from one test-dev run with the chosen settings.
- **Not compared with published results yet.** VisDrone papers use different protocols (splits, input sizes, extra training tricks), so I make no claim to match or beat them.
- **Tiling was not used during training.** The model is trained on whole images and only sees tiles at inference. Training on tiled crops would likely help and is untested.
- **Super-resolution of tiles is untested.** The tile-size result suggests that magnifying objects helps; a learned upscaler is the obvious follow-up.
- **Latency is laptop latency.** Thermal throttling makes timings noisy; the benchmark reports spread (std, p95), not just means.

## License

The code in this repository is released under the [MIT License](LICENSE). Free to use, modify and share.

Third-party pieces keep their own terms:
- **VisDrone2019-DET** (Zhu et al., AISKYEYE team, Tianjin University) is for academic and non-commercial research. Check its terms before reusing the data elsewhere.
- **Ultralytics** (the YOLO library this code depends on) is licensed AGPL-3.0, with an enterprise license available. This repository does not include Ultralytics code or weights, but read its license before using the combination in a commercial product.
