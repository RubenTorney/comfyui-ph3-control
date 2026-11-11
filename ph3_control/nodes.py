from __future__ import annotations

import json


from . import adaln, latent as latent_mod, preprocess, strength as strength_mod, weights
from .patch import RESIDUAL_ON, ROWS, apply_control

CATEGORY = "model/patch/minimax"
CONTROL_TYPES = ["auto", "pose", "depth", "canny"]
CONTROL_TYPE_TIP = ("Informative only: the Union patch infers the kind of control (pose / depth / edges) from the "
                    "image itself.")
RANGE_TIP = "Fraction of the sampling STEPS (0 = first step, 1 = last), not of the sigma range."
V2_TIP = "Experimental, off by default: the v2 patch does not follow the pose on P-H3 (measured). Use the v1 patch."


class PH3ControlSimple:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("MODEL", "LATENT")
    RETURN_NAMES = ("model", "latent")
    FUNCTION = "apply"
    DESCRIPTION = ("MiniMax H3 Fun ControlNet for P-H3-Image, all in one: loads the patch itself, converts the time "
                   "curve, encodes the control image. Plug the P-H3 model, its VAE, a depth / pose / edges image and "
                   "the empty latent: the ControlNet only works when the sampler runs on 2 latent frames, so this "
                   "node returns that latent; after the sampler add 'P-H3 First Frame' before the VAE decode.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The P-H3-Image model (after MiniMaxH3SigmaShift 1.0 / 1.0)."}),
                "vae": ("VAE", {"tooltip": "The P-H3-Image VAE."}),
                "control_image": ("IMAGE", {"tooltip": "Depth map (recommended), shaded puppet, pose or edge map, at the output format."}),
                "control_type": (CONTROL_TYPES, {"default": "auto", "tooltip": CONTROL_TYPE_TIP}),
                "intensity": (strength_mod.CHOICES, {"default": strength_mod.DEFAULT_PRESET,
                                                     "tooltip": "How strictly the result follows the control image."}),
            },
            "optional": {
                "latent": ("LATENT", {"tooltip": "The empty P-H3 latent (PH3ImageConditioning). It comes back as a "
                                      "2-latent-frame latent for the sampler. Leave empty to use 'P-H3 Control Latent' yourself."}),
                "auto_download": ("BOOLEAN", {"default": False, "tooltip": "If the MiniMax H3 patch (2.3 GB) is "
                                  "missing, download it from Hugging Face (no token). Off by default."}),
                "model_patch": ("MODEL_PATCH", {"tooltip": "Optional: a patch loaded with ModelPatchLoader. Empty = loaded "
                                                "by this node from models/model_patches."}),
                "patch_name": ("STRING", {"default": weights.PATCH_NAME, "tooltip": "File name in models/model_patches "
                                          "(ignored when model_patch is plugged). Use the v1 patch."}),
                "strength": ("FLOAT", {"default": strength_mod.DEFAULT_STRENGTH, "min": 0.0, "max": 4.0, "step": 0.01,
                                       "tooltip": "Advanced: only used when intensity = Custom."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                                            "tooltip": "Advanced, only used when intensity = Custom. " + RANGE_TIP}),
                "end_percent": ("FLOAT", {"default": strength_mod.DEFAULT_END, "min": 0.0, "max": 1.0, "step": 0.001,
                                          "tooltip": "Advanced, only used when intensity = Custom. " + RANGE_TIP}),
                "rows": (ROWS, {"default": "first", "tooltip": "Advanced: latent frames that get the correction: the 1st (the image, recommended), all, or the next ones."}),
                "residual_on": (RESIDUAL_ON, {"default": "target", "tooltip": "Advanced: segments that get the correction: the target image only (recommended), with the text, or also the reference images Picture n."}),
                "block_weights": ("STRING", {"default": "1, 1, 1, 1, 1", "tooltip": "Advanced: correction weight along the model depth (5 values, blocks 0, 10, 20, 30, 40). Decreasing weights lose the control."}),
                "allow_v2": ("BOOLEAN", {"default": False, "tooltip": V2_TIP}),
            },
        }

    def apply(self, model, vae, control_image, control_type, intensity, latent=None, auto_download=False,
              model_patch=None, patch_name=weights.PATCH_NAME, strength=strength_mod.DEFAULT_STRENGTH,
              start_percent=0.0, end_percent=strength_mod.DEFAULT_END, rows="first", residual_on="target",
              block_weights="1, 1, 1, 1, 1", allow_v2=False):
        s, a, b = strength_mod.resolve(intensity, strength, start_percent, end_percent)
        out_latent = latent_mod.to_frames(latent, 2) if latent is not None else None
        if s == 0:
            return (model, out_latent)
        if model_patch is None:
            model_patch = weights.load_patch(patch_name or weights.PATCH_NAME, bool(auto_download))
        return (apply_control(model, model_patch, vae, control_image, s, a, b, rows=rows, residual_on=residual_on,
                              block_weights=block_weights, allow_v2=bool(allow_v2)), out_latent)


class PH3ControlAdvanced:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    DESCRIPTION = ("MiniMax H3 Fun ControlNet on P-H3-Image with an explicit model_patch (ModelPatchLoader) and raw "
                   "settings. Needs a 2-latent-frame latent ('P-H3 Control Latent') and 'P-H3 First Frame' after the sampler.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "model_patch": ("MODEL_PATCH",), "vae": ("VAE",), "control_image": ("IMAGE",),
            "strength": ("FLOAT", {"default": strength_mod.DEFAULT_STRENGTH, "min": 0.0, "max": 4.0, "step": 0.01}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": RANGE_TIP}),
            "end_percent": ("FLOAT", {"default": strength_mod.DEFAULT_END, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": RANGE_TIP}),
            "adaln_reference": ("STRING", {"default": adaln.DEFAULT_REFERENCE, "tooltip": "Only used if the shipped "
                                           "adaLN table does not match the model: an H3 fl2va file in models/diffusion_models."}),
        }, "optional": {
            "rows": (ROWS, {"default": "first"}),
            "residual_on": (RESIDUAL_ON, {"default": "target"}),
            "block_weights": ("STRING", {"default": "1, 1, 1, 1, 1"}),
            "allow_v2": ("BOOLEAN", {"default": False, "tooltip": V2_TIP}),
        }}

    def apply(self, model, model_patch, vae, control_image, strength, start_percent, end_percent, adaln_reference,
              rows="first", residual_on="target", block_weights="1, 1, 1, 1, 1", allow_v2=False):
        return (apply_control(model, model_patch, vae, control_image, strength, start_percent, end_percent,
                              adaln_reference, rows, residual_on, block_weights, bool(allow_v2)),)


class PH3ControlLatent:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "make"
    DESCRIPTION = ("Turns the P-H3 image latent (1 latent frame) into a frozen mini-video of 2 latent frames (5 images): "
                   "the H3 Fun ControlNet only works when the base model runs as video. Not needed when you plug the "
                   "latent into 'P-H3 Control (simple)'.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "latent_frames": ("INT", {"default": 2, "min": 1, "max": 7})}}

    def make(self, latent, latent_frames):
        return (latent_mod.to_frames(latent, latent_frames),)


class PH3ControlFirstFrame:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "first"
    DESCRIPTION = "Keeps the first latent frame (the image) of a latent sampled as a mini-video, for the VAE decode."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}

    def first(self, latent):
        return (latent_mod.first_frame(latent),)


class PH3PuppetRender:
    CATEGORY = "image/preprocessors"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    DESCRIPTION = ("Renders a 3D puppet (shaded / depth / edges / normals / mask) from a pose JSON, as control image. "
                   "Pose format: see ph3_control/rig.py and example_workflows/poses/*.json. Runs on CPU, about 1 s.")

    @classmethod
    def INPUT_TYPES(cls):
        from . import puppet3d
        return {"required": {
            "pose_json": ("STRING", {"multiline": True, "default": POSE_EXAMPLE,
                                     "tooltip": '{"angles": {"r_elbow": [60, 0, 0]}, "camera": {"azimuth": 0, "elevation": 0, "distance": 5}}'}),
            "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 32}),
            "height": ("INT", {"default": 1344, "min": 64, "max": 4096, "step": 32}),
            "output": (list(puppet3d.OUTPUTS), {"default": "depth", "tooltip": "depth is the measured default."}),
        }}

    def render(self, pose_json, width, height, output):
        from . import puppet3d
        try:
            pose = json.loads(pose_json)
        except ValueError as e:
            raise ValueError(f"pose_json is not valid JSON: {e}") from e
        return (puppet3d.to_image_tensor(puppet3d.render_pose(pose, width, height)[output]),)


POSE_EXAMPLE = '{"angles": {"r_shoulder": [0, 0, 20], "l_shoulder": [0, 0, 20]}, "camera": {"azimuth": 0, "elevation": 0, "distance": 5}}'


class PH3CannyLight:
    CATEGORY = "image/preprocessors"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    DESCRIPTION = "Edge map (Canny) in pure torch, no extra dependency. White edges on black."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "low_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            "high_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            "blur": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
            "invert": ("BOOLEAN", {"default": False}),
        }}

    def run(self, image, low_threshold, high_threshold, blur, invert):
        if low_threshold > high_threshold:
            low_threshold = high_threshold
        return (preprocess.canny(image, low_threshold, high_threshold, blur, invert),)


class PH3PrepareControlImage:
    CATEGORY = "image/preprocessors"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    DESCRIPTION = "Resize a control image (pose, depth, edges) to the output format, with background and inversion."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "width": ("INT", {"default": 896, "min": 64, "max": 8192, "step": 32}),
            "height": ("INT", {"default": 1120, "min": 64, "max": 8192, "step": 32}),
            "fit": (["fit", "crop", "stretch"], {"default": "fit", "tooltip": "fit: keep ratio and pad; crop: fill and cut; stretch: distort."}),
            "background": (list(preprocess.BACKGROUNDS), {"default": "black"}),
            "invert": ("BOOLEAN", {"default": False}),
        }}

    def run(self, image, width, height, fit, background, invert):
        return (preprocess.prepare(image, width, height, fit, background, invert),)


NODE_CLASS_MAPPINGS = {
    "PH3ControlSimple": PH3ControlSimple,
    "PH3ControlAdvanced": PH3ControlAdvanced,
    "PH3ControlLatent": PH3ControlLatent,
    "PH3ControlFirstFrame": PH3ControlFirstFrame,
    "PH3PuppetRender": PH3PuppetRender,
    "PH3CannyLight": PH3CannyLight,
    "PH3PrepareControlImage": PH3PrepareControlImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PH3ControlSimple": "P-H3 Control (simple)",
    "PH3ControlAdvanced": "P-H3 Control (advanced)",
    "PH3ControlLatent": "P-H3 Control Latent (2 frames)",
    "PH3ControlFirstFrame": "P-H3 First Frame (image)",
    "PH3PuppetRender": "P-H3 Puppet render",
    "PH3CannyLight": "Edges (light Canny)",
    "PH3PrepareControlImage": "Prepare control image",
}
