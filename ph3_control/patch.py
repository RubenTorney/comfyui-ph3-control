"""P-H3 adaptation of the native MiniMax H3 Fun ControlNet patch (ComfyUI >= 0.44).

What differs from the official patch, all measured (6 poses x 2 seeds, PCK@0.1 0.815 vs 0.435 without control):
  1. adaLN 16 -> 8 time-curve conversion (exact affine map, shipped in data/).
  2. The ControlNet only takes effect when the BASE model runs on 2 latent frames (a frozen 5-image video): the
     control image is encoded as a 5-image video (2 latent frames) and the residual is added to the FIRST frame only
     (`rows = first`). Use PH3ControlLatent before the sampler and PH3ControlFirstFrame after it.
  3. The residual goes to the target image tokens only (`residual_on = target`), not to the text nor to the
     reference images Picture n.
  4. start / end are a FRACTION OF THE STEPS (the official sigma conversion is inoperative on the P-H3 grid).
"""
from __future__ import annotations

import logging

import torch

import comfy.model_management
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlPatch

from . import adaln

log = logging.getLogger(__name__)

ROWS = ["first", "all", "rest"]
RESIDUAL_ON = ["target", "target+text", "target+text+refs"]
V1_BLOCKS = 5
V2_MESSAGE = ("MiniMax H3 Fun ControlNet 2.0 is not supported on P-H3-Image (measured: it does not follow the pose, "
              "same result as no control). Use the v1 patch minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors, "
              "or set allow_v2=True (experimental, unvalidated).")


def is_v2(model_patch) -> bool:
    """The v2 patch has 10 control blocks (places 0, 5, ..., 45); v1 has 5."""
    return len(model_patch.model.injection_layers) > V1_BLOCKS


def parse_weights(text: str) -> list[float]:
    try:
        w = [float(v) for v in str(text).replace(";", ",").split(",") if v.strip()]
    except ValueError as e:
        raise ValueError(f"block_weights: numbers separated by commas expected, got {text!r}") from e
    if any(v < 0 for v in w):
        raise ValueError("block_weights must not be negative")
    return w


def step_fraction(grid, sigma: float) -> float:
    """Position of `sigma` in the sampler's sigma grid, as a fraction of the steps (0 = first step, 1 = last)."""
    g = torch.as_tensor(grid).float().flatten().cpu()
    return int((g - sigma).abs().argmin()) / (len(g) - 1)


def step_active(frac: float, start: float, end: float) -> bool:
    """start <= frac < end (end >= 1 means 'until the last step included'; end = 0: never)."""
    if end >= 1.0:
        return frac >= start
    return start <= frac < end


def block_weight(weights: list[float], control_index: int, n_blocks: int) -> float:
    """Weights are spread over the depth: 5 values also serve the 10 blocks of v2."""
    return weights[control_index * len(weights) // n_blocks] if weights else 1.0


def mask_skip(skip: torch.Tensor, segments, audio_pos, lat_t: int, rows: str, residual_on: str) -> torch.Tensor:
    """Zero the residual where it must not be applied (in place). `segments` = layout.segments [(a, b, kind)]."""
    va, vb, _ = segments[-1]
    if rows != "all" and lat_t > 1:
        n = (vb - va) // lat_t
        if rows == "first":
            skip[va + n:vb] = 0
        else:
            skip[va:va + n] = 0
    if residual_on != "target+text+refs":
        for a, b, kind in segments:
            if kind in ("ref_img", "cond") or (kind == "text" and residual_on == "target"):
                skip[a:b] = 0
    skip[audio_pos.to(skip.device)] = 0
    return skip


class PH3FunControlPatch(MiniMaxH3FunControlPatch):
    def __init__(self, *a, adaln_A=None, adaln_c=None, rows="first", residual_on="target", block_weights=None,
                 start_percent=0.0, end_percent=1.0, **k):
        super().__init__(*a, **k)
        if rows not in ROWS or residual_on not in RESIDUAL_ON:
            raise ValueError(f"rows must be in {ROWS} and residual_on in {RESIDUAL_ON}")
        self.adaln_A, self.adaln_c = adaln_A, adaln_c
        self.rows, self.residual_on = rows, residual_on
        self.block_weights = list(block_weights or [])
        self.start_percent, self.end_percent = float(start_percent), float(end_percent)
        self.warned = False

    def _t(self, t_emb: torch.Tensor) -> torch.Tensor:
        if self.adaln_A is None or t_emb.shape[-1] != self.adaln_A.shape[0]:
            return t_emb
        return adaln.convert_t16_to_t8(t_emb, self.adaln_A, self.adaln_c)

    # ---------------------------------------------------------------- control latent
    def _encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        """[T, 3, H, W] in [0, 1] -> video latents [1, 24, T_lat, h, w] (mean, no noise) with the VAE provided: the
        P-H3 image VAE has the same encoder as the H3 video VAE, only its 'image' call differs."""
        vae = self.vae
        x = frames.movedim(0, 1).unsqueeze(0)
        with comfy.model_management.cuda_device_context(vae.device):
            mem = vae.memory_used_encode(x.shape, vae.vae_dtype)
            comfy.model_management.load_models_gpu([vae.patcher], memory_required=mem, force_full_load=vae.disable_offload)
            x = vae.process_input(x).to(vae.vae_dtype)
            z = MiniMaxH3VideoVAE.encode(vae.first_stage_model, x, device=vae.device)
        return z.to(torch.float32)

    @staticmethod
    def frames_needed(lat_t: int) -> int:
        """Images needed for `lat_t` latent frames (1 -> 5 for the first two, then 17 per 5 latent frames)."""
        return max((lat_t - 2) // 5, 0) * 17 + 5 + (17 if (lat_t - 2) % 5 else 0)

    def prepare_control_latent(self, target_shape):
        target_shape = tuple(target_shape)
        if self.mask is not None:
            return super().prepare_control_latent(target_shape)
        if self.control_latent is not None and self.control_latent_shape == target_shape:
            return
        _, _, lat_t, lh, lw = target_shape
        sc = self.vae.spacial_compression_encode()
        loaded = comfy.model_management.loaded_models(only_currently_used=True)
        try:
            if lat_t == 1:       # still-image latent: one image (works, but the control is not followed)
                hint = self._encode(self._fit_frames(self.control_video, 1, lw * sc, lh * sc), target_shape)
            else:
                n = self.frames_needed(lat_t)
                hint = self._encode_video(self._fit_frames(self.control_video, n, lw * sc, lh * sc))[:, :, :lat_t]
                if tuple(hint.shape) != target_shape:
                    raise ValueError(f"control latent {tuple(hint.shape)} unexpected for target {target_shape}")
        finally:
            comfy.model_management.load_models_gpu(loaded)
        self.control_latent = hint
        self.control_latent_shape = target_shape

    # ---------------------------------------------------------------- control branch
    def diffusion_model_wrapper(self, executor, x, timestep, context, transformer_options={}, **kwargs):
        grid = transformer_options.get("sample_sigmas")
        if grid is not None and len(grid) > 1:       # range in fraction of steps
            frac = step_fraction(grid, float(timestep.flatten()[0]) / 1000.0)
            on = step_active(frac, self.start_percent, self.end_percent)
            self.sigma_start, self.sigma_end = (2.0, -1.0) if on else (-1.0, 2.0)
        if x[0].shape[2] == 1 and not self.warned:
            self.warned = True
            log.warning("ph3-control: single-frame latent. The H3 Fun ControlNet is not followed on it and degrades the "
                        "image. Pass the latent through 'P-H3 Control Latent' (2 frames) before the sampler and "
                        "'P-H3 First Frame' after it (the simple node's `latent` input / output does the first part).")
        return super().diffusion_model_wrapper(executor, x, timestep, context, transformer_options, **kwargs)

    def after_block(self, block_index, args, out):
        if not self.active:
            return out
        cn = self.model_patch.model
        t_emb = self._t(args["t_emb"])
        control_index = cn.injection_layers.index(block_index)
        if control_index == 0:
            self.control_latent = self.control_latent.to(out["img"].device)
            self.control_stream = cn.init_stream(self.pristine_stream, self.control_latent, args["layout"], t_emb)
            self.pristine_stream = None
        self.control_stream, skip = cn.step(control_index, self.control_stream, t_emb, args["mod_segments"],
                                            args["rope_freqs"], transformer_options=args["transformer_options"])
        w = block_weight(self.block_weights, control_index, len(cn.injection_layers))
        if w == 0:
            return out
        mask_skip(skip, args["layout"].segments, args["layout"].audio_pos, self.control_latent_shape[2],
                  self.rows, self.residual_on)
        out["img"].add_(skip, alpha=self.strength * w)
        return out


def apply_control(model, model_patch, vae, control_image, strength, start_percent, end_percent,
                  adaln_reference=adaln.DEFAULT_REFERENCE, rows="first", residual_on="target", block_weights="",
                  allow_v2=False):
    if strength == 0:
        return model
    if end_percent < start_percent:
        raise ValueError("end_percent must not be below start_percent")
    if is_v2(model_patch) and not allow_v2:
        raise ValueError(V2_MESSAGE)
    weights = parse_weights(block_weights) if isinstance(block_weights, str) else list(block_weights)
    dm = model.get_model_object("diffusion_model")
    A = c = None
    table = getattr(dm, "adaln_t_table", None)
    need = model_patch.model.control_blocks[0].adaln_proj.linear.in_features
    if table is not None and table.shape[-1] != need:
        A, c, _src = adaln.resolve_map(table, need, adaln_reference)
    patched = model.clone()
    ms = model.get_model_object("model_sampling")
    patch = PH3FunControlPatch(model_patch, vae, control_image[..., :3].movedim(-1, 1), None, None, strength,
                               float(ms.percent_to_sigma(start_percent)), float(ms.percent_to_sigma(end_percent)),
                               adaln_A=A, adaln_c=c, rows=rows, residual_on=residual_on, block_weights=weights,
                               start_percent=start_percent, end_percent=end_percent)
    patch.register(patched)
    return patched
