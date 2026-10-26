"""Load test for the /predict endpoint.

    pip install locust
    IMAGE_DIR=/path/to/VisDrone/images/val \
    locust -f serve/loadtest/locustfile.py --headless -u 8 -r 2 -t 60s \
           --host http://localhost:8000 --csv results/load_c8

Repeat with -u 1, 2, 4, 8, 16 and plot p50/p95/p99 and requests per second against
concurrency. Run against each model (FP32, FP16, INT8) so the table compares like with like,
and note that the single-GPU lock in serve/app.py means throughput saturates early.
Locust's CSV files (--csv) hold the percentiles; copy the numbers you report into the README.
"""
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

IMAGE_DIR = Path(os.environ.get("IMAGE_DIR", "."))
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "50"))
IMAGES = [p.read_bytes() for p in sorted(IMAGE_DIR.glob("*.jpg"))[:MAX_IMAGES]]
if not IMAGES:
    raise SystemExit(f"No .jpg files found in IMAGE_DIR={IMAGE_DIR!s}")


class DetectorUser(HttpUser):
    wait_time = between(0.0, 0.05)

    def _post(self, mode: str):
        self.client.post(
            "/predict", params={"mode": mode},
            files={"file": ("image.jpg", random.choice(IMAGES), "image/jpeg")},
            name=f"/predict?mode={mode}",
        )

    @task(8)
    def full(self):
        self._post("full")

    @task(1)
    def tiled_full(self):
        self._post("tiled_full")
