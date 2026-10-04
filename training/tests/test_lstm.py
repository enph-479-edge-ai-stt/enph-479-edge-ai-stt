"""Tests for what the two models share: the weight image export, run on both of them.

No data, CPU. Skips when torch/torchaudio are absent, which is the case on CI (they
aren't locked deps).
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from training.am.model import AcousticModel  # noqa: E402
from training.cm.model import CharModel  # noqa: E402
from training.lstm.export import export, save_run  # noqa: E402
from training.lstm.train import _quantize, _weight_steps  # noqa: E402


def _unpack(words, width):
    """Signed ``width``-bit values packed side by side in 32-bit words, least significant first."""
    row = sum(word << (32 * k) for k, word in enumerate(words))
    vals = [(row >> s) & (2**width - 1) for s in range(0, 32 * len(words), width)]
    return torch.tensor([v - 2**width if v >= 2 ** (width - 1) else v for v in vals])


@pytest.mark.parametrize("net", [AcousticModel, CharModel], ids=["am", "cm"])
def test_export_packs_the_model_by_the_memory_map(net, tmp_path):
    model = net(n_hidden=32, n_layers=2)
    with pytest.raises(ValueError, match="6-bit grid"):
        export(model, tmp_path, "net")
    _quantize(model, _weight_steps(model), 31)  # what the 6-bit fine-tune leaves behind
    export(model, tmp_path, "net", note="kept")

    # Walk shared/specs/weight_image.md with its numbers written out. A row is PE array 0
    # then PE array 1: 2 x 32 PEs x 6 bits = 12 words.
    words = [int(line, 16) for line in (tmp_path / "net_fabric.mem").read_text().split()]
    rows = iter([words[k : k + 12] for k in range(0, len(words), 12)])

    def take(n, width):
        """The next ``n`` rows as ``[2, n, values per array]``."""
        vals = torch.stack([_unpack(next(rows), width) for _ in range(n)])
        return vals.view(n, 2, -1).transpose(0, 1)

    sd = model.state_dict()
    for layer in range(2):
        w = torch.cat([sd[f"lstm.weight_ih_l{layer}"], sd[f"lstm.weight_hh_l{layer}"]], dim=1)
        b = sd[f"lstm.bias_ih_l{layer}"] + sd[f"lstm.bias_hh_l{layer}"]
        # Per pass: gates i and f, then o and g, as indices into nn.LSTM's i, f, g, o.
        for gates in [(0, 1), (3, 2)]:
            bias = take(4, 24).reshape(2, 32)
            weights = take(w.shape[1], 6).transpose(1, 2)  # [array, PE, input]
            for array, gate in enumerate(gates):
                block = slice(32 * gate, 32 * gate + 32)
                assert torch.equal(bias[array], torch.round(b[block] * 2**14).long())
                assert torch.equal(weights[array], (w[block] * 2**7).long())
    bias = take(4, 24).reshape(2, 32)
    weights = take(32, 6).transpose(1, 2)
    n_out = model.proj.out_features
    assert torch.equal(bias[0, :n_out], torch.round(sd["proj.bias"] * 2**13).long())
    assert torch.equal(weights[0, :n_out], (sd["proj.weight"] * 2**6).long())
    assert not bias[0, n_out:].any() and not weights[0, n_out:].any()
    assert not bias[1].any() and not weights[1].any()
    assert next(rows, None) is None

    manifest = json.loads((tmp_path / "net_fabric.json").read_text())
    assert manifest["rows"] * manifest["words_per_row"] == len(words)
    assert manifest["n_in"] == model.lstm.input_size
    assert manifest["x_frac"] == model.x_frac
    assert manifest["note"] == "kept"


def test_save_run_writes_checkpoints_logs_and_image(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the notebooks write their logs to the working directory
    for log in ["train.log", "train_6bit.log"]:
        (tmp_path / log).write_text("log")
    model = AcousticModel(n_hidden=32, n_layers=1)
    qmodel = AcousticModel(n_hidden=32, n_layers=1)
    _quantize(qmodel, _weight_steps(qmodel), 31)
    mean = torch.arange(123.0)
    save_run(tmp_path / "run", "am", model, qmodel, cmvn_mean=mean)

    run = tmp_path / "run"
    assert sorted(p.name for p in run.iterdir()) == [
        "am.pt",
        "am_6bit.pt",
        "am_fabric.json",
        "am_fabric.mem",
        "train.log",
        "train_6bit.log",
    ]
    for ckpt, m in [("am.pt", model), ("am_6bit.pt", qmodel)]:
        saved = torch.load(run / ckpt)
        assert torch.equal(saved["cmvn_mean"], mean)
        assert torch.equal(saved["model"]["proj.weight"], m.proj.weight)
    assert json.loads((run / "am_fabric.json").read_text())["cmvn_mean"] == mean.tolist()
