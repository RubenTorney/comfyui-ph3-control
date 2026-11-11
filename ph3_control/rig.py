"""Minimal autonomous puppet rig for P-H3 Control (MIT): forward kinematics and pose validation.

Trimmed standalone puppet rig (joint table in data/rig.json). A pose is a JSON object:
{"angles": {joint: [x, y, z] degrees}, "offset": [x, z], "lift": m, "camera": {"azimuth", "elevation", "distance"},
 "extra": [{"angles", "offset"}]}. `extra` holds one optional second subject. Pure Python, no GPU.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
CAMERA_DEFAULT = {"azimuth": 0.0, "elevation": 0.0, "distance": 5.0}
FRAMES = ("full", "upper")
ANGLE_LIMIT = 360.0


@lru_cache(maxsize=1)
def rig() -> dict[str, Any]:
    r = json.loads((HERE / "data" / "rig.json").read_text(encoding="utf-8"))
    names = list(r["joints"])
    r["order"] = names
    seen: set[str] = set()
    for n in names:
        par = r["joints"][n]["parent"]
        assert par is None or par in seen, f"rig.json: parent {par} of {n} defined after it"
        seen.add(n)
    return r


Mat = list[list[float]]
Vec = tuple[float, float, float]


def _mul(a: Mat, b: Mat) -> Mat:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _app(m: Mat, v: Vec) -> Vec:
    return (m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2])


def _rx(a: float) -> Mat:
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def _ry(a: float) -> Mat:
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def _rz(a: float) -> Mat:
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


IDENT: Mat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def local_rotation(name: str, ang: Optional[list[float]]) -> Mat:
    """Rotation locale d'une articulation : Rz(z) · Rx(x) · Ry(y), x multiplié par `fwd` (positif = vers l'avant),
    y et z inversés à droite (poses symétriques avec les mêmes angles)."""
    if not ang:
        return IDENT
    j = rig()["joints"][name]
    x, y, z = (float(v) for v in ang)
    if name.startswith("r_"):
        y, z = -y, -z
    return _mul(_rz(math.radians(z)), _mul(_rx(math.radians(x * j["fwd"])), _ry(math.radians(y))))


CLEARANCE = 0.06            # hauteur de la cheville au-dessus du sol : la pose neutre garde ses repères d'origine
OFFSET_LIMIT = 2.0          # décalage au sol du second sujet (m) ; plus loin, le cadrage rapetisserait les deux sujets


def forward(pose: dict[str, Any], angles: Optional[dict[str, Any]] = None, offset: Optional[list[float]] = None,
            *, ground: bool = True, lift: Optional[float] = None) -> dict[str, Vec]:
    """Positions 3D (m) de toutes les articulations d'un sujet (le principal de la pose, ou `angles` + `offset` donnés).
    Calage au sol : le point le plus bas du corps repose à CLEARANCE du sol (y = 0), sauf `lift` (saut : mètres au-dessus)."""
    if angles is None:
        angles = pose.get("angles") or {}
        offset = pose.get("offset")
        lift = pose.get("lift")
    ox, oz = (offset or [0.0, 0.0])[:2]
    R = rig()
    rot: dict[str, Mat] = {}
    pos: dict[str, Vec] = {}
    for n in R["order"]:
        j = R["joints"][n]
        loc = local_rotation(n, angles.get(n))
        par = j["parent"]
        if par is None:
            rot[n] = loc
            o = j["offset"]
            pos[n] = (o[0] + ox, o[1], o[2] + oz)
        else:
            off = _app(rot[par], tuple(j["offset"]))  # type: ignore[arg-type]
            pos[n] = (pos[par][0] + off[0], pos[par][1] + off[1], pos[par][2] + off[2])
            rot[n] = _mul(rot[par], loc)
    if ground:
        dy = CLEARANCE - min(p[1] for p in pos.values()) + float(lift or 0.0)
        pos = {n: (p[0], p[1] + dy, p[2]) for n, p in pos.items()}
    return pos


def _num(v: Any, what: str) -> float:
    """Nombre fini (refuse bool, texte, NaN, infini, entier démesuré)."""
    if isinstance(v, bool):
        raise ValueError(f"{what}: number expected")
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{what}: number expected")
    if not math.isfinite(f):
        raise ValueError(f"{what}: number expected")
    return f


def normalize(pose: Any) -> dict[str, Any]:
    """Pose nettoyée : seules les articulations connues, angles finis bornés, caméra et cadrage valides.
    Lève ValueError (message anglais court) sur toute forme inattendue (l'API répond alors 400)."""
    if not isinstance(pose, dict):
        raise ValueError("pose must be an object")
    names = rig()["joints"]
    raw = pose.get("angles") or {}
    if not isinstance(raw, dict):
        raise ValueError("angles must be an object")
    angles: dict[str, list[float]] = {}
    for k, v in raw.items():
        if k not in names:
            raise ValueError(f"unknown joint: {k}")
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            raise ValueError(f"angles of {k}: three numbers expected")
        f = [_num(x, f"angles of {k}") for x in v]
        if any(abs(x) > ANGLE_LIMIT for x in f):
            raise ValueError(f"angles of {k}: out of range")
        angles[k] = [round(x, 2) for x in f]
    ex = pose.get("extra")
    if ex is not None and not isinstance(ex, (list, tuple)):
        raise ValueError("extra must be a list")
    extra = []
    for e in (ex or [])[:1]:
        if not isinstance(e, dict):
            raise ValueError("extra subject must be an object")
        sub = normalize({"angles": e.get("angles"), "offset": e.get("offset") if e.get("offset") is not None else [0.9, 0.0],
                         "lift": e.get("lift")})
        extra.append({"angles": sub["angles"], "offset": sub["offset"], "lift": sub["lift"]})
    cam = dict(CAMERA_DEFAULT)
    craw = pose.get("camera")
    if craw is not None and not isinstance(craw, dict):
        raise ValueError("camera must be an object")
    for k, v in (craw or {}).items():
        if k in cam:
            cam[k] = _num(v, f"camera {k}")
    cam["azimuth"] = max(-180.0, min(180.0, cam["azimuth"]))
    cam["elevation"] = max(-60.0, min(80.0, cam["elevation"]))
    cam["distance"] = max(2.5, min(20.0, cam["distance"]))
    frame = pose.get("frame") if pose.get("frame") in FRAMES else "full"
    off = pose.get("offset") if pose.get("offset") is not None else [0.0, 0.0]
    if not isinstance(off, (list, tuple)) or len(off) != 2:
        raise ValueError("offset: two numbers expected")
    off = [round(max(-OFFSET_LIMIT, min(OFFSET_LIMIT, _num(v, "offset"))), 3) for v in off]
    lift = round(max(0.0, min(2.0, _num(pose.get("lift") if pose.get("lift") is not None else 0.0, "lift"))), 3)
    out = {"angles": angles, "offset": off, "lift": lift, "camera": cam, "frame": frame}
    if extra:
        out["extra"] = extra
    return out


def subjects(pose: dict[str, Any]) -> list[dict[str, Vec]]:
    """Positions 3D de chaque sujet de la pose (principal, puis le second s'il existe)."""
    out = [forward(pose)]
    for e in pose.get("extra") or []:
        out.append(forward(pose, e.get("angles") or {}, e.get("offset"), lift=e.get("lift")))
    return out
