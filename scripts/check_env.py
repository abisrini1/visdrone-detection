"""Check that Python, PyTorch (with CUDA) and Ultralytics are set up for training.

    python scripts/check_env.py
"""
import platform
import sys

print(f"Python      {sys.version.split()[0]} ({platform.system()})")

try:
    import torch

    print(f"PyTorch     {torch.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"CUDA        yes | {props.name} | {props.total_memory / 2**30:.1f} GB VRAM")
    else:
        print("CUDA        NO. This PyTorch build cannot see your GPU.")
        print("            Reinstall the CUDA build using the command from https://pytorch.org/get-started/locally/")
except ImportError:
    print("PyTorch     not installed (install the CUDA build first, see README)")

try:
    import ultralytics
    from ultralytics.utils import SETTINGS

    print(f"Ultralytics {ultralytics.__version__}")
    print(f"datasets_dir = {SETTINGS['datasets_dir']}")
    print("            change with: yolo settings datasets_dir=/path/to/datasets")
except ImportError:
    print("Ultralytics not installed (pip install -r requirements.txt)")
