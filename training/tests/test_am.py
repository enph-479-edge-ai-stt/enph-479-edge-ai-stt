"""Acoustic-model tests: features, dataset, collate and model.

Whole runs are tested in test_lstm.py, on both models.

Synthetic data only (noise FLACs in a temp dir), CPU, no network. Skips when
torch/torchaudio are absent, which is the case on CI (they aren't locked deps).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from conftest import TEXTS, write_subset  # noqa: E402

from shared import vocab  # noqa: E402
from training.am.dataset import (  # noqa: E402
    LibriSpeechFeatures,
    collate,
    list_utterances,
    spec_augment,
    speed_perturb,
)
from training.am.model import AcousticModel  # noqa: E402
from training.features import fbank  # noqa: E402


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
    items = list_utterances(write_subset(tmp_path))
    assert [text for _, text in items] == TEXTS
    feats, labels = LibriSpeechFeatures(items)[0]
    assert feats.size(1) == 123
    assert labels.tolist() == vocab.encode("HELLO WORLD")


@pytest.mark.parametrize("speed", [0.9, 1.0, 1.1])
def test_speed_perturb_changes_the_length_by_the_speed(speed):
    wav = torch.randn(1, fbank.SAMPLE_RATE)
    out = speed_perturb(wav, speed)
    assert abs(out.size(1) - fbank.SAMPLE_RATE / speed) <= 1
    assert speed != 1.0 or out is wav


def test_spec_augment_masks_whole_bins_and_frames_but_not_energy():
    torch.manual_seed(0)
    feats = torch.ones(200, fbank.FEATURE_DIM)
    for _ in range(50):
        out = spec_augment(feats)
        assert torch.equal(feats, torch.ones_like(feats))  # a copy, not in place
        cols = (out == 0).all(0)  # zeroed in every frame: a frequency mask
        rows = (out == 0).all(1)  # zeroed in every column: a time mask
        assert not cols[[40, 81, 122]].any()  # the energy columns
        mel = cols[:40]
        assert torch.equal(cols[41:81], mel) and torch.equal(cols[82:122], mel)
        assert mel.sum() <= 14 and rows.sum() <= 2 * 40
        assert torch.equal(out == 0, cols.expand_as(out) | rows[:, None])  # nothing else


def test_dataset_augments_only_when_asked(tmp_path):
    items = list_utterances(write_subset(tmp_path))
    plain = LibriSpeechFeatures(items)
    assert torch.equal(plain[0][0], plain[0][0])
    torch.manual_seed(0)
    aug = LibriSpeechFeatures(items, augment=True)
    reads = [aug[0][0] for _ in range(6)]
    assert any(r.shape != plain[0][0].shape or not torch.equal(r, plain[0][0]) for r in reads)
    assert all(r.size(1) == 123 for r in reads)


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
