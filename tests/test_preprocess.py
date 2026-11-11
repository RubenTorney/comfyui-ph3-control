import torch

from ph3_control import preprocess


def _square():
    img = torch.zeros(1, 64, 64, 3)
    img[:, 16:48, 16:48, :] = 1.0
    return img


def test_canny_finds_square_outline():
    e = preprocess.canny(_square(), 0.1, 0.3, 1.0)
    assert e.shape == (1, 64, 64, 3)
    assert set(e.unique().tolist()) <= {0.0, 1.0}
    on = e[0, :, :, 0] > 0
    assert 60 < int(on.sum()) < 400                         # thin outline (~4 * 32 px), not a filled area
    assert not on[28:36, 28:36].any()                       # interior empty
    assert not on[:8, :].any()                              # far background empty
    assert on[14:19, 20:44].any() and on[20:44, 14:19].any()  # near the edges of the square


def test_canny_flat_image_and_invert():
    flat = torch.full((1, 32, 32, 3), 0.4)
    assert preprocess.canny(flat).sum() == 0
    assert preprocess.canny(flat, invert=True).min() == 1.0
    try:
        preprocess.canny(flat, 0.5, 0.2)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_prepare_fit_crop_stretch_invert():
    img = torch.ones(1, 40, 80, 3)                          # wide white image
    fit = preprocess.prepare(img, 64, 96, "fit", "black")
    assert fit.shape == (1, 96, 64, 3)
    assert fit[0, 0, 0, 0] == 0 and fit[0, 48, 32, 0] == 1  # padded top, image centre
    crop = preprocess.prepare(img, 64, 96, "crop")
    assert crop.shape == (1, 96, 64, 3) and float(crop.min()) > 0.99
    assert preprocess.prepare(img, 64, 96, "stretch").shape == (1, 96, 64, 3)
    inv = preprocess.prepare(img, 80, 40, "fit", "white", invert=True)
    assert float(inv.max()) < 1e-6                          # inverted white image = black
    gray = preprocess.prepare(torch.ones(1, 10, 10, 4), 40, 20, "fit", "gray")   # RGBA input
    assert gray.shape == (1, 20, 40, 3) and abs(float(gray[0, 0, 0, 0]) - 0.5) < 1e-6
