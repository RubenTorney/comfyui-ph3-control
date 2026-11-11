"""Render the control images of a puppet pose, no ComfyUI needed (numpy + Pillow).

python tools/render_puppet.py example_workflows/poses/dance_1.json out/ --size 768x1344
Writes out/<pose>_{depth,shaded,edges,normal,mask}.png
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# load the two modules without importing the package (which needs ComfyUI)
pkg = type(sys)("ph3_control")
pkg.__path__ = [str(ROOT / "ph3_control")]
sys.modules["ph3_control"] = pkg
for name in ("rig", "puppet3d"):
    spec = importlib.util.spec_from_file_location(f"ph3_control.{name}", ROOT / "ph3_control" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
puppet3d = sys.modules["ph3_control.puppet3d"]

ap = argparse.ArgumentParser()
ap.add_argument("pose", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--size", default="768x1344")
a = ap.parse_args()
w, h = (int(v) for v in a.size.lower().split("x"))
a.out.mkdir(parents=True, exist_ok=True)
for kind, im in puppet3d.render_pose(json.loads(a.pose.read_text(encoding="utf-8")), w, h).items():
    im.save(a.out / f"{a.pose.stem}_{kind}.png")
print("ok", a.out)
