import types

import torch

from ph3_control import patch as P


def raises(fn, text):
    try:
        fn()
    except (ValueError, RuntimeError) as e:
        assert text in str(e), str(e)
    else:
        raise AssertionError("expected an error")


def test_step_fraction_and_range():
    grid = torch.tensor([1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0])
    fr = [P.step_fraction(grid, float(s)) for s in grid[:-1]]
    assert fr == [i / 8 for i in range(8)]
    active = [P.step_active(f, 0.0, 0.5) for f in fr]
    assert active == [True] * 4 + [False] * 4                      # steps 0..3 of 8
    assert [P.step_active(f, 0.0, 1.0) for f in fr] == [True] * 8   # end = 1: all steps
    assert [P.step_active(f, 0.25, 0.75) for f in fr] == [False, False, True, True, True, True, False, False]
    assert not any(P.step_active(f, 0.0, 0.0) for f in fr)


def test_wrapper_sets_window_from_grid():
    p = P.PH3FunControlPatch(None, None, torch.zeros(1, 3, 8, 8), None, None, 1.0, 1.0, 0.0, start_percent=0.0, end_percent=0.5)
    grid = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.0])
    seen = []

    # call the window logic through the wrapper with a stub parent
    native = P.MiniMaxH3FunControlPatch.diffusion_model_wrapper
    P.MiniMaxH3FunControlPatch.diffusion_model_wrapper = lambda self, ex, x, t, c, to, **k: seen.append((self.sigma_start, self.sigma_end))
    try:
        for sigma in (1.0, 0.9, 0.8, 0.7):
            p.diffusion_model_wrapper(None, [torch.zeros(1, 1, 2, 1, 1)], torch.tensor([sigma * 1000.0]), None, {"sample_sigmas": grid})
    finally:
        P.MiniMaxH3FunControlPatch.diffusion_model_wrapper = native
    on = [s == (2.0, -1.0) for s in seen]
    assert on == [True, True, False, False]


def test_weights_and_blocks():
    assert P.parse_weights("1, 0.5; 0") == [1.0, 0.5, 0.0]
    assert P.parse_weights("") == []
    raises(lambda: P.parse_weights("a, b"), "block_weights")
    raises(lambda: P.parse_weights("-1"), "negative")
    assert [P.block_weight([1, 2, 3, 4, 5], i, 5) for i in range(5)] == [1, 2, 3, 4, 5]
    assert [P.block_weight([1, 2, 3, 4, 5], i, 10) for i in range(10)] == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert P.block_weight([], 3, 5) == 1.0


def layout(lat_t=2, n=4):
    segs = [(0, 2, "text"), (2, 6, "ref_img"), (6, 6 + lat_t * n, "video")]
    return types.SimpleNamespace(segments=segs, audio_pos=torch.tensor([6 + lat_t * n]))


def rows_hit(skip):
    return [i for i in range(skip.shape[0]) if skip[i].abs().sum() > 0]


def test_residual_on_target_first_frame_only():
    lay = layout()
    skip = torch.ones(6 + 8 + 1, 3)
    P.mask_skip(skip, lay.segments, lay.audio_pos, 2, "first", "target")
    assert rows_hit(skip) == [6, 7, 8, 9]                  # first latent frame of the target only


def test_residual_variants():
    lay = layout()
    def run(rows, res):
        skip = torch.ones(15, 3)
        P.mask_skip(skip, lay.segments, lay.audio_pos, 2, rows, res)
        return rows_hit(skip)
    assert run("all", "target") == list(range(6, 14))
    assert run("rest", "target") == [10, 11, 12, 13]
    assert run("first", "target+text") == [0, 1, 6, 7, 8, 9]
    assert run("first", "target+text+refs") == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]          # audio row 14 always zero


def test_after_block_with_fake_model():
    lay = layout()
    skip_full = torch.ones(15, 3)

    class CN:
        injection_layers = (0, 10)
        def init_stream(self, pristine, latent, layout, t):
            return "s"
        def step(self, i, stream, t, mod, rope, transformer_options=None):
            return stream, skip_full.clone()
    patch = P.PH3FunControlPatch(types.SimpleNamespace(model=CN()), None, torch.zeros(1, 3, 8, 8), None, None, 2.0, 1.0, 0.0,
                                 block_weights=[1, 0.5])
    patch.active = True
    patch.pristine_stream = torch.zeros(15, 3)
    patch.control_latent = torch.zeros(1, 1, 2, 1, 1)
    patch.control_latent_shape = (1, 1, 2, 1, 1)
    args = {"t_emb": torch.zeros(1, 8), "layout": lay, "mod_segments": None, "rope_freqs": None, "transformer_options": {}}
    out = patch.after_block(0, args, {"img": torch.zeros(15, 3)})["img"]
    assert rows_hit(out) == [6, 7, 8, 9] and float(out[6, 0]) == 2.0                  # strength 2.0 x weight 1
    out = patch.after_block(10, args, {"img": torch.zeros(15, 3)})["img"]
    assert float(out[6, 0]) == 1.0                                                   # x weight 0.5
    patch.active = False
    untouched = patch.after_block(0, args, {"img": torch.zeros(15, 3)})["img"]
    assert float(untouched.abs().sum()) == 0


def test_frames_needed():
    assert P.PH3FunControlPatch.frames_needed(2) == 5
    assert P.PH3FunControlPatch.frames_needed(7) == 22 and P.PH3FunControlPatch.frames_needed(3) == 22


def test_v2_refused_and_experimental_option():
    from ph3_control.patch import apply_control
    v2 = types.SimpleNamespace(model=types.SimpleNamespace(injection_layers=tuple(range(0, 50, 5))))
    v1 = types.SimpleNamespace(model=types.SimpleNamespace(injection_layers=(0, 10, 20, 30, 40)))
    assert P.is_v2(v2) and not P.is_v2(v1)
    raises(lambda: apply_control("m", v2, None, torch.zeros(1, 8, 8, 3), 1.2, 0.0, 0.5), "not supported")
    raises(lambda: apply_control("m", v2, None, torch.zeros(1, 8, 8, 3), 1.2, 0.0, 0.5), "v1")
    assert apply_control("m", v2, None, torch.zeros(1, 8, 8, 3), 0.0, 0.0, 0.5) == "m"      # strength 0: nothing to do
    raises(lambda: apply_control("m", v1, None, torch.zeros(1, 8, 8, 3), 1.0, 0.6, 0.2), "end_percent")
    raises(lambda: P.PH3FunControlPatch(None, None, None, None, None, 1.0, 1.0, 0.0, rows="x"), "rows")


def test_single_frame_latent_warns_once():
    import logging
    msgs = []
    h = logging.Handler()
    h.emit = lambda r: msgs.append(r.getMessage())
    P.log.addHandler(h)
    native = P.MiniMaxH3FunControlPatch.diffusion_model_wrapper
    P.MiniMaxH3FunControlPatch.diffusion_model_wrapper = lambda self, *a, **k: "ok"
    try:
        p = P.PH3FunControlPatch(None, None, torch.zeros(1, 3, 8, 8), None, None, 1.0, 1.0, 0.0)
        for _ in range(3):
            p.diffusion_model_wrapper(None, [torch.zeros(1, 1, 1, 1, 1)], torch.tensor([500.0]), None, {})
        q = P.PH3FunControlPatch(None, None, torch.zeros(1, 3, 8, 8), None, None, 1.0, 1.0, 0.0)
        q.diffusion_model_wrapper(None, [torch.zeros(1, 1, 2, 1, 1)], torch.tensor([500.0]), None, {})
    finally:
        P.MiniMaxH3FunControlPatch.diffusion_model_wrapper = native
        P.log.removeHandler(h)
    assert len(msgs) == 1 and "single-frame" in msgs[0]
