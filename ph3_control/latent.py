"""Latent helpers: P-H3 image latent (1 latent frame) <-> frozen mini-video (2 latent frames = 5 images)."""
from __future__ import annotations

import torch

import comfy.nested_tensor


def to_frames(latent: dict, latent_frames: int = 2) -> dict:
    """Empty latent of `latent_frames` latent frames, same spatial size, audio part sized accordingly."""
    video, audio = latent["samples"].unbind()
    if latent_frames == video.shape[2]:
        return latent
    frames = max((latent_frames - 2) // 5, 0) * 17 + 5
    audio_t = max(2, round(frames / 24 * 40)) if latent_frames > 1 else 2
    out = dict(latent)
    out["samples"] = comfy.nested_tensor.NestedTensor((
        torch.zeros(video.shape[:2] + (latent_frames,) + video.shape[3:], dtype=video.dtype, device=video.device),
        torch.zeros(audio.shape[:3] + (audio_t,), dtype=audio.dtype, device=audio.device)))
    return out


def first_frame(latent: dict) -> dict:
    """Keep the first latent frame (the image) for the image VAE."""
    video, audio = latent["samples"].unbind()
    out = dict(latent)
    out["samples"] = comfy.nested_tensor.NestedTensor((video[:, :, :1].contiguous(), audio[..., :2].contiguous()))
    return out
