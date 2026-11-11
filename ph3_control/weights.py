"""Locating, announcing, optionally downloading, and loading the MiniMax H3 Fun ControlNet Union patch."""
from __future__ import annotations

import logging
import os
import shutil

log = logging.getLogger(__name__)

PATCH_NAME = "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors"
PATCH_REPO = "Comfy-Org/MiniMax-H3"
PATCH_REMOTE = f"model_patches/{PATCH_NAME}"
PATCH_SIZE = 2_296_635_360
PATCH_URL = f"https://huggingface.co/{PATCH_REPO}/resolve/main/{PATCH_REMOTE}"
_cache: dict = {}


def human(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def missing_message(name: str = PATCH_NAME) -> str:
    return (f"MiniMax H3 Fun ControlNet patch not found: {name} ({human(PATCH_SIZE)}, MiniMax H3 Community license).\n"
            f"Download it from {PATCH_URL}\n"
            f"and put it in ComfyUI/models/model_patches/ , or tick 'auto_download' on the node.\n"
            f"CLI: huggingface-cli download {PATCH_REPO} {PATCH_REMOTE} --local-dir ComfyUI/models")


def find(name: str = PATCH_NAME):
    import folder_paths
    return folder_paths.get_full_path("model_patches", name)


def announced_size(name: str = PATCH_NAME) -> int:
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        size = get_hf_file_metadata(hf_hub_url(PATCH_REPO, PATCH_REMOTE), token=False).size
        return int(size) if size else PATCH_SIZE
    except Exception:
        return PATCH_SIZE


def download(name: str = PATCH_NAME) -> str:
    """Downloads the official patch (no token) into the first model_patches folder. Only called on explicit opt-in."""
    if name != PATCH_NAME:
        raise RuntimeError(f"auto_download only knows {PATCH_NAME}; put {name} in models/model_patches yourself")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError("huggingface_hub missing.\n" + missing_message(name)) from e
    import folder_paths
    target_dir = folder_paths.get_folder_paths("model_patches")[0]
    os.makedirs(target_dir, exist_ok=True)
    log.warning("ph3-control: downloading %s (%s) from %s", name, human(announced_size(name)), PATCH_REPO)
    tmp = os.path.join(target_dir, ".ph3_download")
    try:
        got = hf_hub_download(PATCH_REPO, PATCH_REMOTE, local_dir=tmp, token=False)
        dest = os.path.join(target_dir, name)
        shutil.move(got, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dest


def ensure(name: str = PATCH_NAME, auto_download: bool = False) -> str:
    path = find(name)
    if path:
        return path
    if not auto_download:
        raise FileNotFoundError(missing_message(name))
    download(name)
    path = find(name)
    if not path:
        raise FileNotFoundError(missing_message(name))
    return path


def load_patch(name: str = PATCH_NAME, auto_download: bool = False):
    """MODEL_PATCH object (cached per file and mtime; one entry, the patch is 2.3 GB)."""
    path = ensure(name, auto_download)
    key = (path, os.path.getmtime(path))
    if _cache.get("key") != key:
        from comfy_extras.nodes_model_patch import ModelPatchLoader
        _cache.clear()
        _cache["key"] = key
        _cache["patch"] = ModelPatchLoader().load_model_patch(name)[0]
    return _cache["patch"]
