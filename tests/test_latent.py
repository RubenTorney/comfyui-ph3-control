import torch

import comfy.nested_tensor


def mk(frames=1):
    return {"samples": comfy.nested_tensor.NestedTensor((torch.randn(1, 24, frames, 4, 3), torch.randn(1, 8, 1, 6))), "extra": 7}


def test_to_two_frames():
    from ph3_control import latent
    out = latent.to_frames(mk(1), 2)
    video, audio = out["samples"].unbind()
    assert tuple(video.shape) == (1, 24, 2, 4, 3) and video.abs().sum() == 0
    assert audio.shape[:3] == (1, 8, 1) and audio.shape[-1] == max(2, round(5 / 24 * 40))
    assert out["extra"] == 7


def test_to_frames_noop_and_more():
    from ph3_control import latent
    same = mk(2)
    assert latent.to_frames(same, 2) is same
    video, _ = latent.to_frames(mk(1), 7)["samples"].unbind()
    assert video.shape[2] == 7


def test_first_frame_selection():
    from ph3_control import latent
    lat = mk(2)
    video, audio = lat["samples"].unbind()
    out = latent.first_frame(lat)
    ov, oa = out["samples"].unbind()
    assert tuple(ov.shape) == (1, 24, 1, 4, 3) and torch.equal(ov[:, :, 0], video[:, :, 0])
    assert oa.shape[-1] == 2 and out["extra"] == 7


def test_nodes_roundtrip():
    from ph3_control.nodes import PH3ControlFirstFrame, PH3ControlLatent
    two = PH3ControlLatent().make(mk(1), 2)[0]
    one = PH3ControlFirstFrame().first(two)[0]
    assert one["samples"].unbind()[0].shape[2] == 1


def test_simple_node_returns_two_frame_latent():
    import ph3_control.nodes as nodes
    from ph3_control import strength
    calls = []
    orig = nodes.apply_control
    nodes.apply_control = lambda *a, **k: calls.append((a[4:7], k)) or "patched"
    try:
        out = nodes.PH3ControlSimple().apply("m", "v", torch.zeros(1, 8, 8, 3), "auto", strength.DEFAULT_PRESET, latent=mk(1),
                                             model_patch=object())
        assert out[0] == "patched" and out[1]["samples"].unbind()[0].shape[2] == 2
        assert calls[0][0] == (1.2, 0.0, 0.5) and calls[0][1]["rows"] == "first" and calls[0][1]["residual_on"] == "target"
        none = nodes.PH3ControlSimple().apply("m", "v", torch.zeros(1, 8, 8, 3), "auto", strength.DEFAULT_PRESET, model_patch=object())
        assert none[1] is None
    finally:
        nodes.apply_control = orig
