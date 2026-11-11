import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")           # tests never touch the GPU
ROOT = Path(__file__).resolve().parents[1]
COMFY = os.environ.get("COMFYUI_PATH") or str(ROOT.parents[1])   # custom_nodes/<pack> -> ComfyUI root
for p in (COMFY, str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from comfy.cli_args import args  # noqa: E402

args.cpu = True                                              # ComfyUI imports without a CUDA device
