"""adaLN time-curve conversion P-H3-Image (16 coords) -> MiniMax H3 video (8 coords).

The official Fun ControlNet patch expects the 8-coordinate time basis of the H3 *video* models; P-H3-Image gives 16.
The H3 curve is an affine combination of the P-H3 one (relative residual ~3e-6): t8 = t16 @ A + c.
A (16x8) and c (8) are shipped precomputed in data/ph3_adaln_map.safetensors, so no 20 GB H3 video model is needed.
If the shipped table does not match the loaded model, and an H3 fl2va file is on disk, the map is refit on the fly.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import torch

log = logging.getLogger(__name__)

DATA = Path(__file__).resolve().parent / "data" / "ph3_adaln_map.safetensors"
DEFAULT_REFERENCE = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
PROBE_TOL = 1e-2       # model table vs shipped fingerprint (the model table may be stored in bf16)
FIT_TOL = 1e-3         # acceptable residual of a refit
_fit_cache: dict = {}


def lerp_table(table: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of an (N, W) table at t in [0, 1]."""
    pos = t.clamp(0.0, 1.0) * (table.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
    return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm().clamp(min=1e-12))


@lru_cache(maxsize=1)
def load_embedded() -> dict | None:
    """Shipped map: {A, c, probe_t, probe_base, probe_target}, or None in a build without the table (--no-table).
    Checks its own integrity (A, c reproduce the target)."""
    from safetensors.torch import load_file
    if not DATA.exists():
        return None
    d = load_file(str(DATA))
    if d["A"].shape[1] != d["c"].shape[0] or d["A"].shape[0] != d["probe_base"].shape[1]:
        raise RuntimeError(f"{DATA.name}: inconsistent shapes")
    res = _rel(d["probe_base"] @ d["A"] + d["c"], d["probe_target"])
    if res > FIT_TOL:
        raise RuntimeError(f"{DATA.name}: corrupted table (residual {res:.2e})")
    return d


def convert_t16_to_t8(t: torch.Tensor, A: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """t_emb (..., 16) -> (..., 8); computed in float32 then cast back to t's dtype."""
    A, c = A.to(t.device), c.to(t.device)
    return (t.float() @ A + c).to(t.dtype)


def embedded_matches(base_table: torch.Tensor) -> bool:
    """Coherence check: does the model's adaLN table look like the one the shipped map was fitted on?"""
    d = load_embedded()
    if base_table.ndim != 2 or base_table.shape[-1] != d["A"].shape[0]:
        return False
    got = lerp_table(base_table.detach().float().cpu(), d["probe_t"])
    return _rel(got, d["probe_base"]) < PROBE_TOL


def _table_from_file(path: str) -> torch.Tensor:
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        key = next(k for k in f.keys() if k.endswith("adaln_t_table"))
        return f.get_tensor(key).float()


def _find_reference(name: str):
    try:
        import folder_paths
        return folder_paths.get_full_path("diffusion_models", name)
    except Exception:
        return None


def fit_map(base_table: torch.Tensor, target_table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Least-squares A, c with target(t) ~ base(t) @ A + c. Returns (A, c, relative residual)."""
    base = base_table.detach().float().cpu()
    x = torch.linspace(0.0, 1.0, target_table.shape[0])
    X = torch.cat([lerp_table(base, x), torch.ones(len(x), 1)], 1).double()
    W = torch.linalg.lstsq(X, target_table.double()).solution
    res = float((X @ W - target_table.double()).norm() / target_table.double().norm())
    return W[:-1].float(), W[-1].float(), res


def resolve_map(base_table: torch.Tensor, need: int, reference_name: str = DEFAULT_REFERENCE):
    """Returns (A, c, source) mapping the model's curve to `need` coordinates, source in {"embedded", "refit", "embedded-unverified"}.

    Order: shipped table if it matches the model; else refit against an H3 video file if present; else the shipped
    table with a warning when the widths at least agree; else a clear error.
    """
    d = load_embedded()
    width_ok = d is not None and d["A"].shape[0] == base_table.shape[-1] and d["A"].shape[1] == need
    if width_ok and embedded_matches(base_table):
        return d["A"], d["c"], "embedded"
    ref = _find_reference(reference_name)
    if ref:
        key = (tuple(base_table.shape), reference_name, ref)
        if key not in _fit_cache:
            target = _table_from_file(ref)
            if target.shape[-1] != need:
                raise RuntimeError(f"{reference_name}: adaLN curve width {target.shape[-1]}, the patch needs {need}")
            A, c, res = fit_map(base_table, target)
            if res > FIT_TOL:
                raise RuntimeError(f"adaLN curve of this model cannot be converted to {reference_name} (residual {res:.2e})")
            log.info("ph3-control: adaLN map refit on %s, residual %.2e", reference_name, res)
            _fit_cache[key] = (A, c)
        return (*_fit_cache[key], "refit")
    if d is None:
        raise RuntimeError(f"this build has no adaLN table: put {reference_name} (the MiniMax H3 fl2va file) in models/diffusion_models "
                           "so the map is computed locally, or build the table once with tools/build_adaln_table.py")
    if width_ok:
        log.warning("ph3-control: this model's adaLN table differs from the one the shipped map was fitted on and no "
                    "reference file %s was found; using the shipped map anyway.", reference_name)
        return d["A"], d["c"], "embedded-unverified"
    raise RuntimeError(f"no adaLN map from width {base_table.shape[-1]} to {need}: put {reference_name} in models/diffusion_models")
