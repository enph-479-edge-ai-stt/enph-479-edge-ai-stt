"""The repack against a small image built straight from shared/specs/weight_image.md."""

import json

import numpy as np
import pytest

from runtime.weights import (
    AM_BASE,
    T_BIAS,
    T_OUT_BIAS,
    T_OUT_WEIGHT,
    T_WEIGHT,
    cm_base,
    read_image,
    repack,
)

H, N_IN, LAYERS, N_OUT = 16, 3, 2, 5
PASSES = ((0, 1), (2, 3))  # gate on PE array 0, gate on PE array 1


def pack(values, width):
    row = 0
    for n, v in enumerate(values):
        row |= (int(v) & (2**width - 1)) << (width * n)
    return [(row >> s) & 0xFFFFFFFF for s in range(0, width * len(values), 32)]


def unpack(words, width, n):
    row = sum(int(w) << (32 * k) for k, w in enumerate(words))
    vals = [(row >> (width * i)) & (2**width - 1) for i in range(n)]
    return [v - 2**width if v >= 2 ** (width - 1) else v for v in vals]


@pytest.fixture
def model():
    rng = np.random.default_rng(0)
    w = [rng.integers(-31, 32, (4, H, (N_IN if layer == 0 else H) + H)) for layer in range(LAYERS)]
    b = [rng.integers(-(2**23), 2**23, (4, H)) for _ in range(LAYERS)]
    wo = rng.integers(-31, 32, (N_OUT, H))
    bo = rng.integers(-(2**23), 2**23, N_OUT)
    return w, b, wo, bo


def image_of(w, b, wo, bo):
    per_row = H * 6 // 24  # biases per array per bias row
    rows = []

    def add_pass(w0, w1, b0, b1):
        for r in range(0, H, per_row):
            rows.append(pack(list(b0[r : r + per_row]) + list(b1[r : r + per_row]), 24))
        for j in range(w0.shape[1]):
            rows.append(pack(list(w0[:, j]) + list(w1[:, j]), 6))

    for layer in range(LAYERS):
        for g0, g1 in PASSES:
            add_pass(w[layer][g0], w[layer][g1], b[layer][g0], b[layer][g1])
    w0, b0 = np.zeros((H, H), int), np.zeros(H, int)
    w0[:N_OUT], b0[:N_OUT] = wo, bo
    add_pass(w0, np.zeros((H, H), int), b0, np.zeros(H, int))
    return np.array(rows, dtype=np.uint32)


MANIFEST = {"n_in": N_IN, "n_hidden": H, "n_layers": LAYERS, "n_out": N_OUT}


def test_every_field_lands_where_the_controller_reads_it(model):
    w, b, wo, bo = model
    loads = {ld.target: ld for ld in repack(image_of(*model), MANIFEST, AM_BASE)}

    wt = loads[T_WEIGHT].words.reshape(-1, 2 * H * 6 // 32)
    r = 0
    for layer in range(LAYERS):
        for g0, g1 in PASSES:
            for j in range(w[layer].shape[2]):
                vals = unpack(wt[r], 6, 2 * H)
                assert vals[:H] == list(w[layer][g0][:, j]) and vals[H:] == list(w[layer][g1][:, j])
                r += 1
    assert loads[T_WEIGHT].rows == r == len(wt)

    bs = loads[T_BIAS].words.reshape(-1, 2 * H * 24 // 32)
    for layer in range(LAYERS):
        for p, (g0, g1) in enumerate(PASSES):
            vals = unpack(bs[2 * layer + p], 24, 2 * H)
            assert vals[:H] == list(b[layer][g0]) and vals[H:] == list(b[layer][g1])
    assert loads[T_BIAS].rows == 2 * LAYERS

    ow = loads[T_OUT_WEIGHT].words.reshape(H, -1)
    assert ow.shape[1] == -(-N_OUT * 6 // 32)
    for h in range(H):
        assert unpack(ow[h], 6, N_OUT) == list(wo[:, h])
    assert loads[T_OUT_WEIGHT].rows == H

    assert unpack(loads[T_OUT_BIAS].words, 24, N_OUT) == list(bo)
    assert len(loads[T_OUT_BIAS].words) == -(-N_OUT * 24 // 32)
    assert loads[T_OUT_BIAS].rows == 1


def test_the_cm_follows_the_am():
    loads = repack(image_of(*_zeros()), MANIFEST, cm_base(MANIFEST))
    am_weight_rows = 2 * (N_IN + H) + (LAYERS - 1) * 4 * H
    assert [ld.addr for ld in loads] == [am_weight_rows, 2 * LAYERS, H, 1]
    assert [ld.addr for ld in repack(image_of(*_zeros()), MANIFEST, AM_BASE)] == [0, 0, 0, 0]


def test_a_wrong_sized_image_is_refused():
    with pytest.raises(ValueError):
        repack(image_of(*_zeros())[:-1], MANIFEST, AM_BASE)


def test_read_image(tmp_path, model):
    image = image_of(*model)
    (tmp_path / "am_fabric.mem").write_text("".join(f"{x:08x}\n" for x in image.ravel()))
    (tmp_path / "am_fabric.json").write_text(json.dumps({**MANIFEST, "rows": len(image)}))
    read, manifest = read_image(tmp_path / "am_fabric.mem")
    assert np.array_equal(read, image) and manifest["rows"] == len(image)


def _zeros():
    w = [np.zeros((4, H, (N_IN if layer == 0 else H) + H), int) for layer in range(LAYERS)]
    return w, [np.zeros((4, H), int)] * LAYERS, np.zeros((N_OUT, H), int), np.zeros(N_OUT, int)
