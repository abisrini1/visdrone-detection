"""Re-encode the tracked video to H.264 so it plays in browsers and on GitHub.

    python -m src.to_web_mp4 --video results/clip_tracked.mp4 --out docs/tracking_demo.mp4

OpenCV writes MPEG-4 Part 2 video, which most browsers cannot play and which is huge.
This converts it to H.264 at 1280 px wide (about 4-5 MB for 15 seconds).

Needs ffmpeg. If `ffmpeg` is not on your PATH, install a bundled copy with:
    pip install imageio-ffmpeg
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from src.utils import REPO_ROOT


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg not found. Run: pip install imageio-ffmpeg")


def build_command(ffmpeg: str, video: str, out: str, width: int = 1280, crf: int = 27):
    return [ffmpeg, "-y", "-loglevel", "error", "-i", str(video),
            "-vf", f"scale={width}:-2", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "tracking_demo.mp4"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--crf", type=int, default=27, help="quality: lower is sharper and bigger (18-32)")
    args = ap.parse_args()

    if not Path(args.video).exists():
        raise SystemExit(f"cannot find {args.video}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_command(find_ffmpeg(), args.video, args.out, args.width, args.crf), check=True)
    print(f"saved {args.out}  ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
