import json
from pathlib import Path

import numpy as np

POSES = Path(__file__).resolve().parents[1] / "example_workflows" / "poses"


def test_render_outputs_and_shapes():
    from ph3_control import puppet3d
    out = puppet3d.render_pose(json.loads((POSES / "stand_front.json").read_text()), 96, 160, supersample=1)
    assert set(out) == {"shaded", "depth", "edges", "normal", "mask"}
    assert {k: (v.mode, v.size) for k, v in out.items()} == {"shaded": ("RGB", (96, 160)), "depth": ("L", (96, 160)),
                                                             "edges": ("L", (96, 160)), "normal": ("RGB", (96, 160)),
                                                             "mask": ("L", (96, 160))}
    mask = np.asarray(out["mask"]) > 127
    assert 0.03 < mask.mean() < 0.5                                          # a body, not empty, not everything
    ys, xs = np.nonzero(mask)
    assert ys.min() > 0 and ys.max() < 159                                   # framed inside the image
    depth = np.asarray(out["depth"])
    assert depth[~mask].max() == 0 and depth[mask].min() > 0                 # black background, body lit by proximity
    assert np.asarray(out["edges"]).max() == 255


def test_all_example_poses_render_and_differ():
    from ph3_control import puppet3d
    masks = {}
    for f in sorted(POSES.glob("*.json")):
        out = puppet3d.render_pose(json.loads(f.read_text()), 64, 96, supersample=1)
        masks[f.stem] = np.asarray(out["mask"])
        assert masks[f.stem].max() == 255
    assert len(masks) >= 6
    assert not np.array_equal(masks["dance_1"], masks["stand_front"])


def test_invalid_pose_refused_and_tensor():
    from ph3_control import puppet3d, rig
    for bad in ({"angles": {"nope": [0, 0, 0]}}, {"angles": {"head": [1, 2]}}, [], {"camera": []}):
        try:
            puppet3d.render_pose(bad, 32, 32)
        except ValueError:
            continue
        raise AssertionError(bad)
    assert rig.forward(rig.normalize({}))["head"][1] > 1.4
    t = puppet3d.to_image_tensor(puppet3d.render_pose({}, 32, 48, supersample=1)["depth"])
    assert tuple(t.shape) == (1, 48, 32, 3) and float(t.max()) <= 1.0


def test_puppet_node():
    from ph3_control.nodes import PH3PuppetRender
    spec = PH3PuppetRender.INPUT_TYPES()["required"]
    assert spec["output"][1]["default"] == "depth"
    pose = json.dumps({"angles": {"r_elbow": [80, 0, 0]}})
    (img,) = PH3PuppetRender().render(pose, 64, 96, "shaded")
    assert tuple(img.shape) == (1, 96, 64, 3)
    try:
        PH3PuppetRender().render("{nope", 64, 96, "depth")
    except ValueError as e:
        assert "JSON" in str(e)
    else:
        raise AssertionError
    json.loads(spec["pose_json"][1]["default"])
