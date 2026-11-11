import asyncio
import json
from pathlib import Path

WF = Path(__file__).resolve().parents[1] / "example_workflows"
NAMES = ["ph3_pose_control_basic", "ph3_pose_control_4mp_hands", "ph3_mannequin_reference"]
_CLASSES = {}


def classes():
    """ComfyUI's own nodes (core + MiniMax H3) plus this pack's."""
    if not _CLASSES:
        import nodes
        asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False))
        from ph3_control.nodes import NODE_CLASS_MAPPINGS
        _CLASSES.update(nodes.NODE_CLASS_MAPPINGS)
        _CLASSES.update(NODE_CLASS_MAPPINGS)
    return _CLASSES


def known_inputs(cls):
    spec = cls.INPUT_TYPES()
    req, opt = set(spec["required"]), set(spec.get("optional", {}))
    return req, opt | {f"images.image_{i}" for i in range(1, 10)} if "images" in opt else opt


def test_api_workflows_match_comfyui_nodes():
    for name in NAMES:
        api = json.loads((WF / f"{name}_api.json").read_text(encoding="utf-8"))
        for nid, n in api.items():
            for v in n["inputs"].values():
                if isinstance(v, list):
                    assert str(v[0]) in api, (name, nid, v)
            cls = classes().get(n["class_type"])
            assert cls is not None, (name, n["class_type"])
            req, opt = known_inputs(cls)
            assert set(n["inputs"]) <= req | opt, (name, n["class_type"], set(n["inputs"]) - req - opt)
            assert req <= set(n["inputs"]), (name, n["class_type"], req - set(n["inputs"]))


def test_ui_workflows_are_consistent():
    for name in NAMES:
        ui = json.loads((WF / f"{name}.json").read_text(encoding="utf-8"))
        by_id = {n["id"]: n for n in ui["nodes"]}
        assert len(by_id) == len(ui["nodes"])
        for lid, o, os_, t, ts, typ in ui["links"]:
            assert o in by_id and t in by_id
            assert lid in by_id[o]["outputs"][os_]["links"]
            assert by_id[t]["inputs"][ts]["link"] == lid
            assert by_id[o]["outputs"][os_]["type"] == typ == by_id[t]["inputs"][ts]["type"]
        assert sum(n["type"] == "Note" for n in ui["nodes"]) == 1
        api = json.loads((WF / f"{name}_api.json").read_text(encoding="utf-8"))
        assert {str(n["id"]) for n in ui["nodes"] if n["type"] != "Note"} == set(api)
        for n in ui["nodes"]:
            assert n["type"] in classes() or n["type"] == "Note", n["type"]


def test_control_workflows_chain():
    for name in NAMES[:2]:
        api = json.loads((WF / f"{name}_api.json").read_text(encoding="utf-8"))
        ctl = next(k for k, n in api.items() if n["class_type"] == "PH3ControlSimple")
        first = next(k for k, n in api.items() if n["class_type"] == "PH3ControlFirstFrame")
        sca = api[first]["inputs"]["latent"][0]
        assert api[sca]["inputs"]["latent_image"] == [ctl, 1]                 # 2-frame latent into the sampler
        assert api[api[sca]["inputs"]["guider"][0]]["inputs"]["model"] == [ctl, 0]
        assert any(n["class_type"] == "VAEDecode" and n["inputs"]["samples"] == [first, 0] for n in api.values())
        from ph3_control.strength import CHOICES
        assert api[ctl]["inputs"]["intensity"] in CHOICES
        assert api[ctl]["inputs"]["patch_name"].endswith("union_pruned_int8_convrot.safetensors")   # v1, not v2


def test_4mp_workflow_controls_directly_at_4mp():
    api = json.loads((WF / "ph3_pose_control_4mp_hands_api.json").read_text(encoding="utf-8"))
    conds = [n for n in api.values() if n["class_type"] == "PH3ImageConditioning"]
    assert len(conds) == 1 and "images.image_1" not in conds[0]["inputs"]          # no upscaling / edit pass
    assert 3.5e6 < conds[0]["inputs"]["width"] * conds[0]["inputs"]["height"] < 5e6
    assert "thumbs-up" in conds[0]["inputs"]["prompt"] and "OK sign" in conds[0]["inputs"]["prompt"]


def test_mannequin_workflow_has_no_controlnet():
    api = json.loads((WF / "ph3_mannequin_reference_api.json").read_text(encoding="utf-8"))
    assert not any(n["class_type"].startswith("PH3Control") for n in api.values())
    cond = next(n for n in api.values() if n["class_type"] == "PH3ImageConditioning")
    assert "<Subject 1>" in cond["inputs"]["prompt"] and "images.image_1" in cond["inputs"]


def test_example_images_exist():
    for name in NAMES:
        api = json.loads((WF / f"{name}_api.json").read_text(encoding="utf-8"))
        for n in api.values():
            if n["class_type"] == "LoadImage":
                assert (WF / "control_images" / n["inputs"]["image"]).stat().st_size > 1000
