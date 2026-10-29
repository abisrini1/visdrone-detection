"""Turn the annotated tracking video into a small GIF for the README.

    python -m src.make_gif --video results/drone_tracked.mp4 --out docs/tracking.gif --seconds 6

GIFs get big quickly, so this downsizes and drops frames: the defaults (640 px wide, 10 fps,
6 seconds) usually give 2-6 MB, which GitHub renders inline. Lower --width or --fps if yours
is larger than ~10 MB.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import REPO_ROOT


def frame_indices(total_frames: int, src_fps: float, start_s: float, seconds: float, out_fps: float):
    """Indices of the source frames to keep for a GIF of `seconds` at `out_fps`."""
    if total_frames <= 0 or src_fps <= 0 or out_fps <= 0:
        return []
    first = int(start_s * src_fps)
    last = min(total_frames, first + int(seconds * src_fps))
    step = src_fps / out_fps
    out, t = [], float(first)
    while int(t) < last:
        out.append(int(t))
        t += step
    return out


def main():
    import cv2
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "tracking.gif"))
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(frame_indices(total, src_fps, args.start, args.seconds, args.fps))
    if not wanted:
        raise SystemExit("no frames selected: check --start / --seconds")

    frames, i = [], 0
    while i <= max(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            h, w = frame.shape[:2]
            scale = args.width / w
            small = cv2.resize(frame, (args.width, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    if not frames:
        raise SystemExit("no frames could be read")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps),
                   loop=0, optimize=True)
    print(f"saved {out}  ({len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
