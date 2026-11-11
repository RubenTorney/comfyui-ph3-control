import torch

from ph3_control import adaln

import os
REF = os.environ.get("PH3_ADALN_REF", "")   # optional reference map (developers)


def test_embedded_shapes():
    d = adaln.load_embedded()
    assert d["A"].shape == (16, 8) and d["c"].shape == (8,)
    assert d["A"].dtype == torch.float32


def test_embedded_matches_reference_map():
    import os
    if not REF or not os.path.exists(REF):
        return                                   # local-only reference, skipped elsewhere
    ref = torch.load(REF)
    d = adaln.load_embedded()
    assert float((d["A"] - ref["A"]).norm() / ref["A"].norm()) < 1e-4
    assert float((d["c"] - ref["c"]).norm() / ref["c"].norm().clamp(min=1e-12)) < 1e-4


def test_convert_t16_to_t8():
    d = adaln.load_embedded()
    t = torch.randn(2, 5, 16, dtype=torch.bfloat16)
    out = adaln.convert_t16_to_t8(t, d["A"], d["c"])
    assert out.shape == (2, 5, 8) and out.dtype == torch.bfloat16
    exp = t.float() @ d["A"] + d["c"]
    assert torch.allclose(out.float(), exp, rtol=1e-2, atol=1e-2)


def test_embedded_reproduces_target_curve():
    d = adaln.load_embedded()
    res = (d["probe_base"] @ d["A"] + d["c"] - d["probe_target"]).norm() / d["probe_target"].norm()
    assert float(res) < 1e-4


def test_coherence_check():
    d = adaln.load_embedded()
    # a model table whose curve equals the fingerprint: 3 rows linear interpolation of probes is enough to match
    base = d["probe_base"].clone()
    assert adaln.embedded_matches(base)
    assert not adaln.embedded_matches(base * 3 + 1)
    assert not adaln.embedded_matches(torch.randn(100, 8))      # wrong width


def test_resolve_map_embedded_and_errors():
    d = adaln.load_embedded()
    A, c, src = adaln.resolve_map(d["probe_base"].clone(), 8)
    assert src == "embedded" and A.shape == (16, 8)
    A, c, src = adaln.resolve_map(d["probe_base"] * 3 + 1, 8, "does_not_exist.safetensors")
    assert src == "embedded-unverified"
    try:
        adaln.resolve_map(torch.randn(10, 12), 8, "does_not_exist.safetensors")
    except RuntimeError as e:
        assert "no adaLN map" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_fit_map_recovers_affine():
    base = torch.rand(64, 16)
    A0, c0 = torch.randn(16, 8), torch.randn(8)
    target = base @ A0 + c0
    A, c, res = adaln.fit_map(base, target)
    assert res < 1e-5 and torch.allclose(A, A0, atol=1e-3)


def test_build_without_table_gives_a_clear_error():
    saved = adaln.DATA
    adaln.load_embedded.cache_clear()
    adaln.DATA = saved.with_name("absent.safetensors")
    try:
        try:
            adaln.resolve_map(torch.zeros(8, 16), 8, "no_such_file.safetensors")
        except RuntimeError as e:
            assert "no adaLN table" in str(e)
        else:
            raise AssertionError("expected a RuntimeError")
    finally:
        adaln.DATA = saved
        adaln.load_embedded.cache_clear()
