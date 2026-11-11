"""Computes ph3_control/data/ph3_adaln_map.safetensors locally (only needed for a build made with --no-table).

python tools/build_adaln_table.py P-H3-IMAGE.safetensors H3_FL2VA.safetensors [output]
Reads only the `adaln_t_table` tensor of each file (a few MB), not the 20 GB of weights. The output defaults to
ph3_control/data/ in this pack. Without the table the pack also works if the H3 fl2va file is in models/diffusion_models.
"""
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PROBES = 33


def table(path):
    with safe_open(path, "pt") as f:
        key = next(k for k in f.keys() if k.endswith("adaln_t_table"))
        return f.get_tensor(key).float()


def lerp(tab, t):
    pos = t.clamp(0, 1) * (tab.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=tab.shape[0] - 2)
    return torch.lerp(tab[i0], tab[i0 + 1], (pos - i0).unsqueeze(1))


def main():
    base, target = table(sys.argv[1]), table(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).resolve().parents[1] / "ph3_control/data/ph3_adaln_map.safetensors"
    x = torch.linspace(0, 1, target.shape[0])
    X = torch.cat([lerp(base, x), torch.ones(len(x), 1)], 1).double()
    W = torch.linalg.lstsq(X, target.double()).solution
    res = float((X @ W - target.double()).norm() / target.double().norm())
    print("résidu relatif", res)
    pt = torch.linspace(0, 1, PROBES)
    save_file({"A": W[:-1].float().contiguous(), "c": W[-1].float().contiguous(), "probe_t": pt,
               "probe_base": lerp(base, pt).contiguous(), "probe_target": lerp(target, pt).contiguous()},
              str(out), metadata={"residual": f"{res:.3e}", "base_rows": str(base.shape[0]),
                                  "target_rows": str(target.shape[0]), "format": "ph3-adaln-map-1"})
    print("écrit", out)


if __name__ == "__main__":
    main()
