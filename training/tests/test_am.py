"""Acoustic-model tests: features, dataset, collate, model, training loop, and FPGA export.

Synthetic data only (noise FLACs in a temp dir), CPU, no network. Skips when
torch/torchaudio are absent, which is the case on CI (they aren't locked deps).
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

import soundfile as sf  # noqa: E402

from training import vocab  # noqa: E402
from training.am.dataset import (  # noqa: E402
    LibriSpeechFeatures,
    collate,
    compute_cmvn_over,
    list_utterances,
)
from training.am.export import export  # noqa: E402
from training.am.model import AcousticModel  # noqa: E402
from training.am.train import TrainConfig, train  # noqa: E402
from training.features import fbank  # noqa: E402

TEXTS = ["HELLO WORLD", "CAT", "IT'S FINE", "DOG"]


def _write_subset(root):
    """Four noise FLACs (0.5-1.1 s) with transcripts, in LibriSpeech's layout."""
    chapter = root / "1" / "2"
    chapter.mkdir(parents=True)
    lines = []
    for i, text in enumerate(TEXTS):
        wav = (torch.randn(8000 + 3000 * i) * 0.1).numpy()
        sf.write(str(chapter / f"1-2-{i:04d}.flac"), wav, 16000, format="FLAC")
        lines.append(f"1-2-{i:04d} {text}")
    (chapter / "1-2.trans.txt").write_text("\n".join(lines), encoding="utf-8")
    return root


def test_extract_is_123_dim_at_100_frames_per_s():
    mel, deltas = fbank.build_transforms()
    feats = fbank.extract(torch.randn(1, fbank.SAMPLE_RATE), mel, deltas)  # 1 s
    assert feats.shape == (1 + (fbank.SAMPLE_RATE - fbank.N_FFT) // fbank.HOP_LENGTH, 123)


def test_cmvn_normalizes_to_zero_mean_unit_std():
    feats = [torch.randn(50, fbank.FEATURE_DIM) * 3 + 7 for _ in range(4)]
    mean, std = fbank.compute_cmvn(feats)
    normed = torch.cat([fbank.apply_cmvn(f, mean, std) for f in feats])
    assert torch.allclose(normed.mean(0), torch.zeros(fbank.FEATURE_DIM), atol=1e-4)
    assert torch.allclose(normed.std(0, unbiased=False), torch.ones(fbank.FEATURE_DIM), atol=1e-2)


def test_dataset_reads_flac_and_encodes_labels(tmp_path):
    items = list_utterances(_write_subset(tmp_path))
    assert [text for _, text in items] == TEXTS
    feats, labels = LibriSpeechFeatures(items)[0]
    assert feats.size(1) == 123
    assert labels.tolist() == vocab.encode("HELLO WORLD")


def test_collate_pads_and_packs_lengths():
    batch = [
        (torch.randn(10, fbank.FEATURE_DIM), torch.tensor(vocab.encode("HI"))),
        (torch.randn(7, fbank.FEATURE_DIM), torch.tensor(vocab.encode("YES"))),
    ]
    feats, feat_len, labels, label_len = collate(batch)
    assert feats.shape == (2, 10, fbank.FEATURE_DIM)
    assert feat_len.tolist() == [10, 7]
    assert label_len.tolist() == [2, 3]
    assert labels.numel() == 5


def test_model_outputs_log_probs_per_frame():
    model = AcousticModel(n_hidden=32, n_layers=1)
    logp = model(torch.randn(3, 20, fbank.FEATURE_DIM), torch.tensor([20, 15, 12]))
    assert logp.shape == (3, 20, vocab.VOCAB_SIZE)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(3, 20), atol=1e-4)


def test_overfit_one_batch_drives_loss_down():
    """The real loss/gradient/CTC wiring on two sequences a tiny LSTM can memorize."""
    torch.manual_seed(0)
    model = AcousticModel(n_hidden=32, n_layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ctc = torch.nn.CTCLoss(blank=vocab.BLANK_IDX, zero_infinity=True)
    feats = torch.randn(2, 30, fbank.FEATURE_DIM)
    feat_len = torch.tensor([30, 30])
    labels = torch.tensor(vocab.encode("CAT") + vocab.encode("DOG"))
    label_len = torch.tensor([3, 3])
    losses = []
    for _ in range(120):
        loss = ctc(model(feats, feat_len).transpose(0, 1), labels, feat_len, label_len)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.4 * losses[0]


def test_train_runs_end_to_end_on_tiny_data(tmp_path):
    items = list_utterances(_write_subset(tmp_path))
    mean, std = compute_cmvn_over(items)
    cfg = TrainConfig(n_hidden=16, n_layers=1, batch_size=2, epochs=1, num_workers=0, device="cpu")
    datasets = {  # two train datasets, to check they get pooled
        "train": [
            LibriSpeechFeatures(items[:2], mean, std),
            LibriSpeechFeatures(items[2:], mean, std),
        ],
        "test": [LibriSpeechFeatures(items[:2], mean, std)],
    }
    model = train(cfg, datasets, tmp_path / "train.log")
    assert isinstance(model, AcousticModel)
    assert "[epoch 0]" in (tmp_path / "train.log").read_text(encoding="utf-8")


def test_train_with_weight_bits_returns_quantized_weights(tmp_path):
    items = list_utterances(_write_subset(tmp_path))
    mean, std = compute_cmvn_over(items)
    cfg = TrainConfig(
        n_hidden=16, n_layers=1, batch_size=2, epochs=1, num_workers=0, device="cpu", weight_bits=6
    )
    datasets = {
        "train": [LibriSpeechFeatures(items, mean, std)],
        "test": [LibriSpeechFeatures(items[:2], mean, std)],
    }
    init_state = AcousticModel(n_hidden=16, n_layers=1).state_dict()
    model = train(cfg, datasets, tmp_path / "train.log", init_state)
    for name, w in model.named_parameters():
        if "weight" in name:  # 6 bits: integers in [-31, 31] times one step per matrix
            assert w.unique().numel() <= 63, name


def _unpack(words, width):
    """Signed ``width``-bit values packed side by side in 32-bit words, least significant first."""
    row = sum(word << (32 * k) for k, word in enumerate(words))
    vals = [(row >> s) & (2**width - 1) for s in range(0, 32 * len(words), width)]
    return torch.tensor([v - 2**width if v >= 2 ** (width - 1) else v for v in vals])


def test_export_packs_the_model_by_the_memory_map(tmp_path):
    items = list_utterances(_write_subset(tmp_path))
    mean, std = compute_cmvn_over(items)
    cfg = TrainConfig(
        n_hidden=32, n_layers=2, batch_size=2, epochs=1, num_workers=0, device="cpu", weight_bits=6
    )
    datasets = {
        "train": [LibriSpeechFeatures(items, mean, std)],
        "test": [LibriSpeechFeatures(items[:2], mean, std)],
    }
    float_model = AcousticModel(n_hidden=32, n_layers=2)
    with pytest.raises(ValueError, match="6-bit grid"):
        export(float_model, mean, std, tmp_path)
    model = train(cfg, datasets, tmp_path / "train.log", float_model.state_dict())
    export(model, mean, std, tmp_path)

    # Walk shared/specs/am_weight_image.md with its numbers written out. A row is PE array 0
    # then PE array 1: 2 x 32 PEs x 6 bits = 12 words.
    words = [int(line, 16) for line in (tmp_path / "am_fabric.mem").read_text().split()]
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
    n_out = vocab.VOCAB_SIZE
    assert torch.equal(bias[0, :n_out], torch.round(sd["proj.bias"] * 2**13).long())
    assert torch.equal(weights[0, :n_out], (sd["proj.weight"] * 2**6).long())
    assert not bias[0, n_out:].any() and not weights[0, n_out:].any()
    assert not bias[1].any() and not weights[1].any()
    assert next(rows, None) is None

    manifest = json.loads((tmp_path / "am_fabric.json").read_text())
    assert manifest["rows"] * manifest["words_per_row"] == len(words)
    assert manifest["cmvn_mean"] == mean.tolist()
