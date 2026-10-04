"""Tests for what the two models share, run on both: a whole run, and the weight image.

Synthetic data only, CPU, no network. Skips when torch/torchaudio are absent, which is
the case on CI (they aren't locked deps).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from conftest import SENTENCES, write_subset  # noqa: E402

from training.am import train as am  # noqa: E402
from training.am.dataset import (  # noqa: E402
    LibriSpeechFeatures,
    compute_cmvn_over,
    list_utterances,
)
from training.am.model import AcousticModel  # noqa: E402
from training.cm import train as cm  # noqa: E402
from training.cm.dataset import to_stream  # noqa: E402
from training.cm.model import CharModel  # noqa: E402
from training.lstm.export import export, save_run  # noqa: E402
from training.lstm.model import LstmNet  # noqa: E402
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


@pytest.fixture(params=["am", "cm"])
def tiny(request, tmp_path):
    """Each model's train module, data small enough to train in a test, and settings to match."""
    if request.param == "am":
        items = list_utterances(write_subset(tmp_path / "audio"))
        mean, std = compute_cmvn_over(items)
        data = {  # two train datasets, to check they get pooled
            "train": [
                LibriSpeechFeatures(items[:2], mean, std),
                LibriSpeechFeatures(items[2:], mean, std),
            ],
            "test": [LibriSpeechFeatures(items[:2], mean, std)],
        }
        return am, data, {"batch_size": 2, "num_workers": 0}
    data = {"train": to_stream(SENTENCES * 10), "test": to_stream(SENTENCES)}
    return cm, data, {"batch_size": 2, "bptt": 10}


def test_a_run_trains_fine_tunes_and_saves(tiny, tmp_path, monkeypatch):
    """A launcher notebook's train, fine-tune and save steps, for each model."""
    net, data, settings = tiny
    assert callable(net.load_data)  # the notebooks' data step; it downloads, so not run here
    monkeypatch.chdir(tmp_path)  # the notebooks write their logs to the working directory
    cfg = net.TrainConfig(n_hidden=32, n_layers=2, epochs=4, device="cpu", **settings)
    qcfg = cfg.fine_tune()  # a tenth of the learning rate, a quarter of the epochs, 6 bits
    assert (qcfg.lr, qcfg.epochs, qcfg.weight_bits) == (pytest.approx(cfg.lr / 10), 1, 6)

    model = net.train(cfg, data, "train.log")
    qmodel = net.train(qcfg, data, "train_6bit.log", model.state_dict())
    assert isinstance(model, LstmNet)
    log = Path("train.log").read_text(encoding="utf-8")
    assert "[epoch 3]" in log
    assert net is am or "sample:" in log  # the CM prints a sampled line every epoch
    for name, w in qmodel.named_parameters():
        if "weight" in name:  # 6 bits: integers in [-31, 31] times one step per matrix
            assert w.unique().numel() <= 63, name

    extra = torch.arange(123.0)
    save_run("run", "net", model, qmodel, cmvn_mean=extra)
    assert sorted(p.name for p in Path("run").iterdir()) == [
        "net.pt",
        "net_6bit.pt",
        "net_fabric.json",
        "net_fabric.mem",
        "train.log",
        "train_6bit.log",
    ]
    for ckpt, m in [("net.pt", model), ("net_6bit.pt", qmodel)]:
        saved = torch.load(Path("run") / ckpt)
        assert torch.equal(saved["cmvn_mean"], extra)
        assert torch.equal(saved["model"]["proj.weight"], m.proj.weight)
    assert json.loads(Path("run/net_fabric.json").read_text())["cmvn_mean"] == extra.tolist()
