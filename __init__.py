"""ControlNet MiniMax H3 Fun for P-H3-Image (ComfyUI custom node pack). See README.md."""
import logging

try:
    from .ph3_control.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError as e:                      # ComfyUI too old: no native MiniMax H3 classes
    logging.getLogger(__name__).error("comfyui-ph3-control needs ComfyUI >= 0.44 (MiniMax H3): %s", e)
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}

WEB_DIRECTORY = None
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
