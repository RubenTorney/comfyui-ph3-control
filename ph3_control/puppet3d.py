"""3D puppet to control images (shaded, depth, edges, normals, mask), numpy + Pillow only, no GPU.

Analytic ray tracing (per-pixel z-buffer) of a classic puppet: capsule bones, joint balls, ellipsoid head, hands and
feet, Lambert material, one directional light plus ambient. Pose in, images out:

    from ph3_control import puppet3d
    out = puppet3d.render_pose(pose_dict, 768, 1344)      # pose_dict: see ph3_control/rig.py
    out["depth"].save("depth.png")                        # PIL images: shaded, depth, edges, normal, mask

Camera frame: x right, y up, z toward the camera, camera at (0, 0, distance). Depth: near = bright, black background,
1 - (z - near) / (far - near) with near = d - 1.6 m, far = d + 1.6 m. `View.fit` frames the subjects full-frame.
MIT, Ruben Torney.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from PIL import Image

Vec = Sequence[float]

# (articulation a, articulation b, rayon m) — table BONES de PoseEditor3D.tsx
BONES = [
    ("pelvis", "spine", 0.1), ("spine", "chest", 0.105), ("chest", "neck", 0.075), ("neck", "head", 0.04),
    ("chest", "l_shoulder", 0.05), ("chest", "r_shoulder", 0.05), ("pelvis", "l_hip", 0.07), ("pelvis", "r_hip", 0.07),
    ("l_shoulder", "l_elbow", 0.045), ("l_elbow", "l_wrist", 0.036), ("r_shoulder", "r_elbow", 0.045), ("r_elbow", "r_wrist", 0.036),
    ("l_hip", "l_knee", 0.07), ("l_knee", "l_ankle", 0.052), ("r_hip", "r_knee", 0.07), ("r_knee", "r_ankle", 0.052),
]
JOINT_BALLS = [("l_shoulder", 0.052), ("r_shoulder", 0.052), ("l_elbow", 0.04), ("r_elbow", 0.04), ("l_wrist", 0.034), ("r_wrist", 0.034),
               ("l_hip", 0.072), ("r_hip", 0.072), ("l_knee", 0.056), ("r_knee", 0.056), ("l_ankle", 0.05), ("r_ankle", 0.05),
               ("pelvis", 0.105), ("chest", 0.108)]
HEAD_RADII = (0.095, 0.125, 0.105)            # largeur, hauteur, profondeur (m)
HAND_LEN, HAND_R = 0.10, 0.032                # main : prolonge l'avant-bras (volumes.py)
FOOT_LEN, FOOT_R = 0.15, 0.042                # pied : vers l'avant du bassin, un peu vers le bas
BODY_COLOR = (0xC4, 0xC4, 0xC4)
LIGHT = (0.5, 0.9, 1.0)                       # lumière clé de l'éditeur (repère caméra)
DEPTH_HALF_RANGE = 1.6                        # near = d − 1,6 ; far = d + 1,6 (shader de profondeur de l'éditeur)


# ------------------------------------------------------------------------------------------------ caméra
@dataclass
class View:
    """Camera of a framing: rotation around `centre` (azimuth, elevation in degrees), perspective d/(d - z), then
    scale `s` (px per metre at z = 0) and frame centre (cx, cy)."""
    azimuth: float
    elevation: float
    distance: float
    centre: tuple[float, float, float]
    s: float
    cx: float
    cy: float

    @classmethod
    def fit(cls, people: list[dict[str, Vec]], width: int, height: int, *, azimuth: float = 0.0, elevation: float = 0.0,
            distance: float = 5.0, margin: float = 0.06, pad: float = 0.11) -> "View":
        """Cadrage plein cadre de tous les sujets (marge relative `margin`, volumes gonflés de `pad` m)."""
        centre = (float(np.mean([p["pelvis"][0] for p in people])), float(people[0]["pelvis"][1]), float(np.mean([p["pelvis"][2] for p in people])))
        v = cls(azimuth, elevation, distance, centre, 1.0, 0.0, 0.0)
        pts = np.array([q for p in people for q in list(p.values()) + [t for t, _ in _tips(p)]], dtype=np.float64)
        cam = v.to_camera(pts)
        k = distance / np.maximum(0.5, distance - cam[:, 2])
        x, y, r = cam[:, 0] * k, cam[:, 1] * k, pad * k
        x0, x1, y0, y1 = (x - r).min(), (x + r).max(), (y - r).min(), (y + r).max()
        v.s = float(min(width * (1 - 2 * margin) / max(x1 - x0, 1e-3), height * (1 - 2 * margin) / max(y1 - y0, 1e-3)))
        v.cx, v.cy = float((x0 + x1) / 2), float((y0 + y1) / 2)
        return v

    def to_camera(self, p: Any) -> np.ndarray:
        """Points monde [..., 3] → repère caméra (rig._view)."""
        p = np.asarray(p, dtype=np.float64) - np.asarray(self.centre, dtype=np.float64)
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        x, z = x * math.cos(-az) + z * math.sin(-az), -x * math.sin(-az) + z * math.cos(-az)
        y, z = y * math.cos(el) - z * math.sin(el), y * math.sin(el) + z * math.cos(el)
        return np.stack([x, y, z], axis=-1)

    def to_pixel(self, cam: np.ndarray, width: int, height: int, scale: float = 1.0) -> tuple[float, float, float]:
        """Point caméra → (x px, y px, px par mètre à sa profondeur)."""
        k = self.distance / max(0.5, self.distance - float(cam[2]))
        s = self.s * scale
        return (width / 2 + (cam[0] * k - self.cx) * s, height / 2 - (cam[1] * k - self.cy) * s, s * k)


# ------------------------------------------------------------------------------------------------ pantin → primitives
def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])


def _tips(p: dict[str, Vec]) -> list[tuple[np.ndarray, bool]]:
    """Bout des mains et des pieds (monde) : (point, « haut du corps »)."""
    P = {k: np.asarray(v, dtype=np.float64) for k, v in p.items()}
    out = [(P[f"{s}_wrist"] + _unit(P[f"{s}_wrist"] - P[f"{s}_elbow"]) * HAND_LEN, True) for s in "lr"]
    ax = P["l_hip"] - P["r_hip"]
    face = _unit(np.array([-ax[2], 0.0, ax[0]]))                    # devant du bassin (+Z pour un sujet de face)
    out += [(P[f"{s}_ankle"] + face * FOOT_LEN + np.array([0.0, -0.05, 0.0]), False) for s in "lr"]
    return out


def primitives(p: dict[str, Vec]) -> list[tuple]:
    """Primitives monde d'un sujet : ("capsule", a, b, r) | ("sphere", c, r) | ("ellipsoid", c, axes 3×3 en colonnes, rayons)."""
    P = {k: np.asarray(v, dtype=np.float64) for k, v in p.items()}
    out: list[tuple] = [("capsule", P[a], P[b], r) for a, b, r in BONES if a in P and b in P]
    out += [("sphere", P[j], r) for j, r in JOINT_BALLS if j in P]
    up = _unit(P["head_top"] - P["head"])
    side = P["l_ear"] - P["r_ear"] if "l_ear" in P and "r_ear" in P else P["l_shoulder"] - P["r_shoulder"]
    side = _unit(side - up * float(side @ up))
    out.append(("ellipsoid", (P["head"] + P["head_top"]) / 2, np.stack([side, up, np.cross(side, up)], axis=1), HEAD_RADII))
    tips = _tips(p)
    for s, (tip, _) in zip("lr", tips[:2]):
        out.append(("capsule", P[f"{s}_wrist"], tip, HAND_R))
    for s, (toe, _) in zip("lr", tips[2:]):
        out.append(("capsule", P[f"{s}_ankle"], toe, FOOT_R))
    return out


# ------------------------------------------------------------------------------------------------ lancer de rayons
def _hit_sphere(O, D, c, r):
    oc = O - c
    b = D @ oc
    h = b * b - (oc @ oc - r * r)
    t = np.where(h >= 0, -b - np.sqrt(np.maximum(h, 0)), np.inf)
    n = (O + D * t[:, None] - c) / r
    return t, n


def _hit_capsule(O, D, a, b, r):
    ba, oa = b - a, O - a
    baba, bard, baoa = float(ba @ ba), D @ ba, float(oa @ ba)
    if baba < 1e-10:
        return _hit_sphere(O, D, a, r)
    rdoa, oaoa = D @ oa, float(oa @ oa)
    A = baba - bard * bard
    B = baba * rdoa - baoa * bard
    C = baba * oaoa - baoa * baoa - r * r * baba
    h = B * B - A * C
    t = np.where(h >= 0, (-B - np.sqrt(np.maximum(h, 0))) / np.maximum(A, 1e-12), np.inf)
    y = baoa + t * bard
    body = (h >= 0) & (y > 0) & (y < baba)
    t1, _ = _hit_sphere(O, D, a, r)
    t2, _ = _hit_sphere(O, D, b, r)
    t = np.where(body, t, np.minimum(t1, t2))
    p = O + D * np.where(np.isfinite(t), t, 0.0)[:, None]
    u = np.clip(((p - a) @ ba) / baba, 0.0, 1.0)
    n = (p - (a + ba * u[:, None])) / r
    return t, n


def _hit_ellipsoid(O, D, c, axes, radii):
    rad = np.asarray(radii, dtype=np.float64)
    o = ((O - c) @ axes) / rad
    d = (D @ axes) / rad
    A, B, C = (d * d).sum(1), d @ o, float(o @ o) - 1.0
    h = B * B - A * C
    t = np.where(h >= 0, (-B - np.sqrt(np.maximum(h, 0))) / A, np.inf)
    pl = o + d * np.where(np.isfinite(t), t, 0.0)[:, None]
    n = (pl / rad) @ axes.T
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return t, n


def render(people: Iterable[dict[str, Vec]], view: View, width: int, height: int, *, supersample: int = 2,
           background: tuple[int, int, int] = (128, 128, 128), color: tuple[int, int, int] = BODY_COLOR,
           light: Vec = LIGHT, ambient: float = 0.38, diffuse: float = 0.62, specular: float = 0.08,
           ground: bool = False, ground_color: tuple[int, int, int] = (150, 150, 150),
           depth_range: Optional[tuple[float, float]] = None, edge_width: float = 1.6) -> dict[str, Image.Image]:
    """Rend le pantin (un ou deux sujets). `ground` ajoute le sol (y = 0) à l'ombré et à la profondeur (dégradé au sol,
    utile en profondeur : les pieds sont posés). Voir l'en-tête pour les sorties."""
    ss = max(1, int(supersample))
    W, H = width * ss, height * ss
    d = view.distance
    s = view.s * ss
    zbuf = np.full((H, W), np.inf)
    nbuf = np.zeros((H, W, 3))
    ibuf = np.zeros((H, W), dtype=np.int32)                       # 0 fond, 1 sol, 2+ primitives
    O = np.array([0.0, 0.0, d])

    def rays(x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
        xs = (np.arange(x0, x1) + 0.5 - W / 2) / s + view.cx
        ys = -(np.arange(y0, y1) + 0.5 - H / 2) / s + view.cy
        X, Y = np.meshgrid(xs, ys)
        D = np.stack([X, Y, np.full_like(X, -d)], axis=-1).reshape(-1, 3)
        return D / np.linalg.norm(D, axis=1, keepdims=True)

    def put(x0: int, x1: int, y0: int, y1: int, t: np.ndarray, n: np.ndarray, ident: int) -> None:
        t = t.reshape(y1 - y0, x1 - x0)
        win = zbuf[y0:y1, x0:x1]
        m = t < win
        win[m] = t[m]
        nbuf[y0:y1, x0:x1][m] = n.reshape(y1 - y0, x1 - x0, 3)[m]
        ibuf[y0:y1, x0:x1][m] = ident

    if ground:
        gp = view.to_camera(np.array([view.centre[0], 0.0, view.centre[2]]))
        gn = view.to_camera(np.array([view.centre[0], 1.0, view.centre[2]])) - gp
        D = rays(0, W, 0, H)
        den = D @ gn
        t = np.where(den < -1e-6, ((gp - O) @ gn) / np.where(den < -1e-6, den, 1.0), np.inf)
        t = np.where((t > 0) & (t < d + 12.0), t, np.inf)
        put(0, W, 0, H, t, np.broadcast_to(gn, D.shape), 1)

    ident = 2
    for person in people:
        for prim in primitives(person):
            kind = prim[0]
            if kind == "capsule":
                ends, r = [view.to_camera(prim[1]), view.to_camera(prim[2])], prim[3]
            elif kind == "sphere":
                ends, r = [view.to_camera(prim[1])], prim[2]
            else:
                ends, r = [view.to_camera(prim[1])], max(prim[3])
            box = []
            for e in ends:
                px, py, k = view.to_pixel(e, W, H, ss)
                box.append((px - r * k * 1.25 - 2, px + r * k * 1.25 + 2, py - r * k * 1.25 - 2, py + r * k * 1.25 + 2))
            x0, x1 = max(0, int(min(b[0] for b in box))), min(W, int(max(b[1] for b in box)) + 1)
            y0, y1 = max(0, int(min(b[2] for b in box))), min(H, int(max(b[3] for b in box)) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            D = rays(x0, x1, y0, y1)
            if kind == "capsule":
                t, n = _hit_capsule(O, D, ends[0], ends[1], r)
            elif kind == "sphere":
                t, n = _hit_sphere(O, D, ends[0], r)
            else:
                axes = view.to_camera(prim[2].T + np.asarray(view.centre)).T          # axes (colonnes) tournés dans le repère caméra
                t, n = _hit_ellipsoid(O, D, ends[0], axes, prim[3])
            put(x0, x1, y0, y1, np.where(t > 0, t, np.inf), n, ident)
            ident += 1

    hit = np.isfinite(zbuf)
    body = ibuf >= 2
    D = rays(0, W, 0, H).reshape(H, W, 3)
    zcam = d - (O + D * np.where(hit, zbuf, 0.0)[..., None])[..., 2]                 # distance le long de l'axe optique
    # ombré : Lambert + ambiante (+ un léger reflet), couleur du pantin ; sol mat
    L = np.asarray(light, dtype=np.float64)
    L = L / np.linalg.norm(L)
    ndl = np.clip(nbuf @ L, 0.0, 1.0)
    half = L - D
    half /= np.linalg.norm(half, axis=-1, keepdims=True)
    spec = np.clip((nbuf * half).sum(-1), 0.0, 1.0) ** 32 * specular
    lum = ambient + diffuse * ndl
    shaded = np.empty((H, W, 3))
    shaded[:] = np.asarray(background, dtype=np.float64)
    shaded[body] = np.clip(np.asarray(color, dtype=np.float64) * lum[body][:, None] / (ambient + diffuse) + 255.0 * spec[body][:, None], 0, 255)
    if ground:
        g = ibuf == 1
        fade = np.clip((zcam[g] - d) / 10.0, 0.0, 1.0)[:, None]                      # le sol se fond dans le fond au loin
        shaded[g] = np.asarray(ground_color, dtype=np.float64) * (1 - fade) + np.asarray(background, dtype=np.float64) * fade
    # profondeur : loi du shader three.js
    near, far = depth_range or (max(0.2, d - DEPTH_HALF_RANGE), d + DEPTH_HALF_RANGE)
    depth = np.where(hit, 1.0 - np.clip((zcam - near) / (far - near), 0.0, 1.0), 0.0)
    # contours : changement de primitive avec saut de profondeur, ou silhouette
    zc = np.where(hit, zcam, 1e3)
    edge = np.zeros((H, W), dtype=bool)
    for ax in (0, 1):
        dz = np.abs(np.diff(zc, axis=ax)) > 0.035
        di = np.diff(ibuf, axis=ax) != 0
        sil = np.diff(body.astype(np.int8), axis=ax) != 0
        e = (dz & di) | sil
        if ax == 0:
            edge[1:] |= e
            edge[:-1] |= e
        else:
            edge[:, 1:] |= e
            edge[:, :-1] |= e
    edges = Image.fromarray((edge * 255).astype(np.uint8))
    grow = max(1, int(round(edge_width * ss)))
    if grow > 1:
        from PIL import ImageFilter
        edges = edges.filter(ImageFilter.MaxFilter(grow | 1))
    normal = np.where(hit[..., None], (nbuf * 0.5 + 0.5) * 255.0, 128.0)

    def down(a: np.ndarray, mode: str) -> Image.Image:
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), mode)
        return im.resize((width, height), Image.LANCZOS) if ss > 1 else im

    return {"shaded": down(shaded, "RGB"), "depth": down(depth * 255.0, "L"), "normal": down(normal, "RGB"),
            "edges": edges.resize((width, height), Image.LANCZOS) if ss > 1 else edges, "mask": down(body * 255.0, "L")}


def render_pose(pose: dict, width: int, height: int, **kw) -> dict[str, Image.Image]:
    """Validate a pose dict (ph3_control.rig), frame it full-frame with its own camera, render all control images."""
    from . import rig
    p = rig.normalize(pose)
    people = rig.subjects(p)
    cam = p["camera"]
    view = View.fit(people, width, height, azimuth=cam["azimuth"], elevation=cam["elevation"], distance=cam["distance"])
    return render(people, view, width, height, **kw)


OUTPUTS = ("depth", "shaded", "edges", "normal", "mask")


def to_image_tensor(img: Image.Image):
    """PIL image -> ComfyUI IMAGE tensor [1, H, W, 3] float in [0, 1]."""
    import torch
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a)[None]
