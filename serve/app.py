"""FastAPI service around the VisDrone detector.

    MODEL_PATH=exports/s_960_960_trt-fp16.engine IMGSZ=960 uvicorn serve.app:app --port 8000

    curl -F "file=@some_image.jpg" "http://localhost:8000/predict?mode=full"
    curl -F "file=@some_image.jpg" "http://localhost:8000/predict?mode=tiled_full"

Endpoints
    GET  /health    200 once the model is loaded, 503 before that
    POST /predict   multipart image -> boxes, scores, classes, per-stage timings
    GET  /metrics   Prometheus metrics (latency histogram, request counter)

Modes: `full` (one pass), `tiled` (overlapping native-resolution tiles) and `tiled_full`
(tiles plus the whole image, merged). The mode is spelled `tiled_full`, not `tiled+full`,
because a literal "+" in a query string decodes to a space.

Settings come from environment variables (see Settings below), so the same image runs any
exported model. Inference is serialized with a lock: one GPU model instance is not safe to
call from several threads, and concurrency is measured by the load test, not hidden here.
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel

from src.tiling import predict_full, predict_tiled
from src.utils import parse_device

try:  # metrics are optional so the unit tests and a bare install still work
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    REQUESTS = Counter("detector_requests_total", "Prediction requests", ["mode", "status"])
    LATENCY = Histogram(
        "detector_request_latency_ms", "End-to-end /predict latency in ms", ["mode"],
        buckets=(5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000),
    )
except ImportError:  # pragma: no cover
    REQUESTS = LATENCY = None

VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
Mode = Literal["full", "tiled", "tiled_full"]


@dataclass
class Settings:
    model_path: str = ""
    imgsz: int = 960
    device: str = "0"
    half: bool = False          # only applies to .pt weights; exported engines bake in precision
    max_det: int = 300
    tile: int = 640
    overlap: float = 0.25
    merge_metric: str = "ios"
    merge_thr: float = 0.6
    tile_batch: int = 16
    max_upload_mb: int = 20

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            model_path=env("MODEL_PATH", ""),
            imgsz=int(env("IMGSZ", "960")),
            device=env("DEVICE", "0"),
            half=env("HALF", "0") == "1",
            max_det=int(env("MAX_DET", "300")),
            tile=int(env("TILE", "640")),
            overlap=float(env("OVERLAP", "0.25")),
            merge_metric=env("MERGE_METRIC", "ios"),
            merge_thr=float(env("MERGE_THR", "0.6")),
            tile_batch=int(env("TILE_BATCH", "16")),
            max_upload_mb=int(env("MAX_UPLOAD_MB", "20")),
        )


class Detection(BaseModel):
    box: list[float]  # x1, y1, x2, y2 in original image pixels
    score: float
    class_id: int
    label: str


class Timings(BaseModel):
    decode_ms: float
    inference_ms: float
    total_ms: float


class PredictResponse(BaseModel):
    mode: str
    image_width: int
    image_height: int
    num_detections: int
    detections: list[Detection]
    timings_ms: Timings


def load_model(settings: Settings):
    """Load the weights/engine and run one dummy pass so the first request is not the slow one."""
    from ultralytics import YOLO

    if not settings.model_path:
        raise RuntimeError("MODEL_PATH is not set")
    model = YOLO(settings.model_path, task="detect")
    dummy = np.zeros((settings.imgsz, settings.imgsz, 3), dtype=np.uint8)
    predict_full(model, dummy, imgsz=settings.imgsz, conf=0.25, iou=0.7, max_det=settings.max_det,
                 device=parse_device(settings.device), half=_use_half(settings))
    return model


def _use_half(settings: Settings) -> bool:
    return settings.half and settings.model_path.endswith(".pt")


def _label(model, class_id: int) -> str:
    names = getattr(model, "names", None) or VISDRONE_CLASSES
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return str(names[class_id]) if 0 <= class_id < len(names) else str(class_id)


def create_app(model=None, settings: Settings | None = None, load_from_env: bool = True) -> FastAPI:
    """Build the app. Tests pass a fake `model` and `load_from_env=False`."""
    settings = settings or Settings.from_env()
    state = {"model": model, "lock": threading.Lock(), "settings": settings}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["model"] is None and load_from_env:
            state["model"] = load_model(settings)
        yield

    app = FastAPI(title="VisDrone small-object detector", lifespan=lifespan)
    app.state.service = state

    @app.get("/health")
    def health():
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {"status": "ok", "model": settings.model_path or "injected", "imgsz": settings.imgsz,
                "device": settings.device}

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        if LATENCY is None:
            raise HTTPException(status_code=501, detail="prometheus_client is not installed")
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict", response_model=PredictResponse)
    def predict(
        file: UploadFile = File(...),
        mode: Mode = Query("full"),
        conf: float = Query(0.25, ge=0.0, le=1.0),
        iou: float = Query(0.7, ge=0.0, le=1.0),
    ):
        t_start = time.perf_counter()
        status = "error"
        try:
            if state["model"] is None:
                raise HTTPException(status_code=503, detail="model not loaded")

            limit = settings.max_upload_mb * 2**20
            data = file.file.read(limit + 1)
            if len(data) > limit:
                raise HTTPException(status_code=413, detail=f"image larger than {settings.max_upload_mb} MB")
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise HTTPException(status_code=400, detail="could not decode the upload as an image")
            t_decoded = time.perf_counter()

            device = parse_device(settings.device)
            half = _use_half(settings)
            with state["lock"]:
                t_infer = time.perf_counter()
                if mode == "full":
                    boxes, scores, classes = predict_full(
                        state["model"], image, imgsz=settings.imgsz, conf=conf, iou=iou,
                        max_det=settings.max_det, device=device, half=half)
                else:
                    boxes, scores, classes = predict_tiled(
                        state["model"], image, tile=settings.tile, overlap=settings.overlap,
                        imgsz=settings.imgsz, conf=conf, iou=iou, max_det=settings.max_det,
                        include_full=(mode == "tiled_full"), merge_metric=settings.merge_metric,
                        merge_thr=settings.merge_thr, device=device, half=half,
                        tile_batch=settings.tile_batch)
                t_done = time.perf_counter()

            detections = [
                Detection(box=[float(v) for v in b], score=float(s), class_id=int(c),
                          label=_label(state["model"], int(c)))
                for b, s, c in zip(boxes, scores, classes)
            ]
            total_ms = (time.perf_counter() - t_start) * 1000.0
            status = "ok"
            return PredictResponse(
                mode=mode, image_width=image.shape[1], image_height=image.shape[0],
                num_detections=len(detections), detections=detections,
                timings_ms=Timings(decode_ms=round((t_decoded - t_start) * 1000.0, 2),
                                   inference_ms=round((t_done - t_infer) * 1000.0, 2),
                                   total_ms=round(total_ms, 2)),
            )
        finally:
            if REQUESTS is not None:
                REQUESTS.labels(mode=mode, status=status).inc()
                if status == "ok":
                    LATENCY.labels(mode=mode).observe((time.perf_counter() - t_start) * 1000.0)

    return app


app = create_app()
