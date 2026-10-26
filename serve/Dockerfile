# Build from the repo root:   docker build -f serve/Dockerfile -t visdrone-serve .
# Run:   docker run --gpus all -p 8000:8000 -v "$PWD/exports:/models:ro" \
#            -e MODEL_PATH=/models/s_960_960_trt-fp16_dyn16.engine -e IMGSZ=960 visdrone-serve
#
# The base image ships PyTorch and TensorRT built together. RTX 50-series (Blackwell) GPUs
# need a tag with CUDA 12.8 or newer, so check the NVIDIA NGC catalog and override
# BASE_IMAGE with a current tag if this one is too old for your driver/GPU.
# A TensorRT .engine is tied to the TensorRT version and GPU it was built on: build engines
# inside this same image (docker run ... python -m serve.export_dynamic ...), not on the host.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3
FROM ${BASE_IMAGE}

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY serve/requirements-serve.txt serve/requirements-serve.txt
RUN pip install -r serve/requirements-serve.txt

COPY src src
COPY serve serve
COPY configs configs

ENV MODEL_PATH=/models/model.engine IMGSZ=960 DEVICE=0
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

# One worker: a single GPU model instance, one process, requests serialized by a lock.
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
