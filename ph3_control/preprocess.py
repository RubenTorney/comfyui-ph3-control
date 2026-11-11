"""Light, dependency-free control-image preprocessing (pure torch). Images are ComfyUI IMAGE: (B, H, W, C) float 0..1."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

BACKGROUNDS = {"black": 0.0, "gray": 0.5, "white": 1.0}


def _gray(img: torch.Tensor) -> torch.Tensor:
    rgb = img[..., :3]
    w = torch.tensor([0.299, 0.587, 0.114], dtype=rgb.dtype, device=rgb.device)
    return (rgb * w).sum(-1, keepdim=True).movedim(-1, 1)           # (B,1,H,W)


def _gauss(sigma: float, dtype, device) -> torch.Tensor:
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=dtype, device=device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def canny(image: torch.Tensor, low: float = 0.1, high: float = 0.3, blur: float = 1.0, invert: bool = False) -> torch.Tensor:
    """Canny edges in pure torch: Gaussian blur, Sobel, non-maximum suppression, hysteresis. White edges on black
    (inverted: black on white). low/high are fractions of the strongest gradient, 0..1."""
    if not 0 <= low <= high <= 1:
        raise ValueError("canny: need 0 <= low <= high <= 1")
    g = _gray(image).float()
    if blur > 0:
        k = _gauss(blur, g.dtype, g.device)
        r = k.numel() // 2
        g = F.conv2d(F.pad(g, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
        g = F.conv2d(F.pad(g, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=g.dtype, device=g.device).view(1, 1, 3, 3)
    gp = F.pad(g, (1, 1, 1, 1), mode="replicate")
    gx, gy = F.conv2d(gp, sx), F.conv2d(gp, sx.transpose(2, 3))
    mag = torch.hypot(gx, gy)
    peak = mag.amax(dim=(1, 2, 3), keepdim=True).clamp(min=0.05)   # flat image: no edges (Sobel of a 0..1 image peaks at 4)
    mag = mag / peak
    # non-maximum suppression along the gradient direction, quantised to 4 directions
    ang = (torch.atan2(gy, gx) * 180 / math.pi) % 180
    mp = F.pad(mag, (1, 1, 1, 1))
    def sh(dy, dx):
        return mp[:, :, 1 + dy:1 + dy + mag.shape[2], 1 + dx:1 + dx + mag.shape[3]]
    d0 = (ang < 22.5) | (ang >= 157.5)
    d45 = (ang >= 22.5) & (ang < 67.5)
    d90 = (ang >= 67.5) & (ang < 112.5)
    n1 = torch.where(d0, sh(0, 1), torch.where(d45, sh(-1, 1), torch.where(d90, sh(1, 0), sh(1, 1))))
    n2 = torch.where(d0, sh(0, -1), torch.where(d45, sh(1, -1), torch.where(d90, sh(-1, 0), sh(-1, -1))))
    thin = mag * ((mag >= n1) & (mag >= n2))
    strong = thin >= high
    weak = thin >= low
    edges = strong
    for _ in range(32):                                              # hysteresis: grow strong edges through weak ones
        grown = (F.max_pool2d(edges.float(), 3, 1, 1) > 0) & weak
        if torch.equal(grown, edges):
            break
        edges = grown
    out = edges.float().movedim(1, -1).repeat(1, 1, 1, 3)
    return 1.0 - out if invert else out


def prepare(image: torch.Tensor, width: int, height: int, fit: str = "fit", background: str = "black",
            invert: bool = False) -> torch.Tensor:
    """Resize a control image to the output size. fit: 'fit' (keep ratio, pad with background), 'crop' (fill, centre
    crop) or 'stretch'. invert: 1 - image (applied before padding, so the background stays as chosen)."""
    if fit not in ("fit", "crop", "stretch"):
        raise ValueError(f"unknown fit mode: {fit!r}")
    if background not in BACKGROUNDS:
        raise ValueError(f"unknown background: {background!r}")
    img = image[..., :3].movedim(-1, 1).float()
    if invert:
        img = 1.0 - img
    b, _, h, w = img.shape
    if fit == "stretch":
        scale_w, scale_h = width, height
    else:
        k = min(width / w, height / h) if fit == "fit" else max(width / w, height / h)
        scale_w, scale_h = max(1, round(w * k)), max(1, round(h * k))
    if (scale_w, scale_h) != (w, h):
        img = F.interpolate(img, size=(scale_h, scale_w), mode="bilinear", align_corners=False,
                            antialias=scale_w < w or scale_h < h).clamp(0, 1)
    if fit == "crop":
        y0, x0 = (scale_h - height) // 2, (scale_w - width) // 2
        img = img[:, :, y0:y0 + height, x0:x0 + width]
    elif fit == "fit":
        canvas = torch.full((b, 3, height, width), BACKGROUNDS[background], dtype=img.dtype, device=img.device)
        y0, x0 = (height - scale_h) // 2, (width - scale_w) // 2
        canvas[:, :, y0:y0 + scale_h, x0:x0 + scale_w] = img
        img = canvas
    return img.movedim(1, -1).contiguous()
