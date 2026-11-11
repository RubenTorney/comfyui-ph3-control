import torch

from ph3_control import strength


class FakeVAE:
    calls = []

    def spacial_compression_encode(self):
        return 16

    def encode(self, x):                                   # (B, H, W, C)
        FakeVAE.calls.append(tuple(x.shape))
        b, h, w, _ = x.shape
        return torch.full((1, 4, b, h // 16, w // 16), 0.5)


class FakePatchModel:
    pass


def test_prepare_control_latent_single_image():
    from ph3_control.patch import PH3FunControlPatch
    FakeVAE.calls.clear()
    img = torch.rand(1, 3, 64, 48)                         # control_video is (B, C, H, W)
    p = PH3FunControlPatch(FakePatchModel(), FakeVAE(), img, None, None, 1.0, 1.0, 0.0)
    shape = (1, 4, 1, 4, 3)                                # 1 latent frame, 64x48 px
    p.prepare_control_latent(shape)
    assert FakeVAE.calls == [(1, 64, 48, 3)]               # ONE image encoded (the native patch would encode 5)
    assert tuple(p.control_latent.shape) == shape and p.control_latent_shape == shape
    p.prepare_control_latent(shape)                        # cached: no second encode
    assert len(FakeVAE.calls) == 1


def test_t_conversion_in_patch():
    from ph3_control import adaln
    from ph3_control.patch import PH3FunControlPatch
    d = adaln.load_embedded()
    p = PH3FunControlPatch(None, None, None, None, None, 1.0, 1.0, 0.0, adaln_A=d["A"], adaln_c=d["c"])
    assert p._t(torch.zeros(1, 16)).shape == (1, 8)
    assert p._t(torch.zeros(1, 8)).shape == (1, 8)         # already 8: untouched


def test_strength_presets():
    assert strength.resolve("Light (Légère) 1.2 · 25 %", 9, 0.3, 0.4) == (1.2, 0.0, 0.25)
    assert strength.resolve("Normal (Normale) 1.2 · 50 %", 9, 0, 1) == (1.2, 0.0, 0.5)
    assert strength.resolve("Strict (Stricte) 1.2 · 75 %", 9, 0, 1) == (1.2, 0.0, 0.75)
    assert "Very strict (Très stricte) 1.8" not in strength.PRESETS        # mesuré moins bon : pas de préréglage
    assert strength.resolve(strength.CUSTOM, 2.0, 0.1, 0.7) == (2.0, 0.1, 0.7)
    assert strength.DEFAULT_PRESET == "Normal (Normale) 1.2 · 50 %" and (strength.DEFAULT_STRENGTH, strength.DEFAULT_END) == (1.2, 0.5)
    for bad in (lambda: strength.resolve("nope", 1, 0, 1), lambda: strength.resolve(strength.CUSTOM, 1, 0.8, 0.2)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    assert strength.DEFAULT_PRESET in strength.CHOICES


def test_nodes_registered():
    from ph3_control.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    assert set(NODE_CLASS_MAPPINGS) == {"PH3ControlSimple", "PH3ControlAdvanced", "PH3ControlLatent", "PH3ControlFirstFrame",
                                          "PH3PuppetRender", "PH3CannyLight", "PH3PrepareControlImage"}
    assert set(NODE_DISPLAY_NAME_MAPPINGS) == set(NODE_CLASS_MAPPINGS)
    for name, cls in NODE_CLASS_MAPPINGS.items():
        info = cls.INPUT_TYPES()
        assert "required" in info and callable(getattr(cls, cls.FUNCTION)) and cls.RETURN_TYPES, name
    req = NODE_CLASS_MAPPINGS["PH3ControlSimple"].INPUT_TYPES()["required"]
    assert list(req) == ["model", "vae", "control_image", "control_type", "intensity"]
    opt = NODE_CLASS_MAPPINGS["PH3ControlSimple"].INPUT_TYPES()["optional"]
    assert opt["auto_download"][1]["default"] is False and opt["allow_v2"][1]["default"] is False
    assert opt["strength"][1]["default"] == 1.2 and opt["end_percent"][1]["default"] == 0.5
    assert opt["rows"][1]["default"] == "first" and opt["residual_on"][1]["default"] == "target"
    assert opt["block_weights"][1]["default"] == "1, 1, 1, 1, 1"
    assert NODE_CLASS_MAPPINGS["PH3ControlSimple"].RETURN_TYPES == ("MODEL", "LATENT")


def test_root_package_loads():
    import importlib.util
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("comfyui_ph3_control", root / "__init__.py",
                                                  submodule_search_locations=[str(root)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["comfyui_ph3_control"] = mod
    spec.loader.exec_module(mod)
    assert "PH3ControlSimple" in mod.NODE_CLASS_MAPPINGS


def test_missing_patch_message_and_no_download():
    from ph3_control import weights
    msg = weights.missing_message()
    assert "huggingface.co/Comfy-Org/MiniMax-H3" in msg and "auto_download" in msg and "MiniMax H3 Community" in msg
    try:
        weights.ensure("absent_patch.safetensors", auto_download=False)
    except FileNotFoundError as e:
        assert "auto_download" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")
    try:
        weights.download("other.safetensors")
    except RuntimeError as e:
        assert "auto_download only knows" in str(e)


def test_apply_control_end_to_end_with_fakes():
    import types

    import comfy.model_patcher
    from ph3_control import adaln
    from ph3_control.patch import PH3FunControlPatch, apply_control

    d = adaln.load_embedded()

    class Sampling:
        def percent_to_sigma(self, p):
            return 1.0 - p

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_model = types.SimpleNamespace(adaln_t_table=d["probe_base"].clone())
            self.model_sampling = Sampling()
            self.model_config = types.SimpleNamespace(unet_config={})
            self.model_dtype = lambda: torch.float32
            self.current_patcher = None
            self.model_options = {}

    def mk_model():
        m = comfy.model_patcher.ModelPatcher(Net(), torch.device("cpu"), torch.device("cpu"))
        return m

    block = types.SimpleNamespace(adaln_proj=types.SimpleNamespace(linear=types.SimpleNamespace(in_features=8)))
    mp = types.SimpleNamespace(model=types.SimpleNamespace(control_blocks=[block], injection_layers=[3, 7]))
    base = mk_model()
    img = torch.rand(1, 32, 32, 3)
    assert apply_control(base, mp, FakeVAE(), img, 0.0, 0.0, 1.0) is base          # strength 0: unchanged
    out = apply_control(base, mp, FakeVAE(), img, 0.8, 0.0, 1.0)
    assert out.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 3)].control_patch.rows == "first"
    assert out is not base
    keys = out.model_options["transformer_options"]["patches_replace"]["dit"]
    assert ("double_block", 3) in keys and ("double_block", 7) in keys
    patch = keys[("double_block", 3)].control_patch
    assert isinstance(patch, PH3FunControlPatch) and patch.adaln_A.shape == (16, 8) and patch.strength == 0.8
