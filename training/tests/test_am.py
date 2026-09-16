"""Acoustic-model tests: model, features, collate, and an overfit wiring check.

All synthetic (no audio files, no network) and CPU-only, but they need the ML
stack, so the whole module skips when torch/torchaudio are absent (e.g. a lean
CI runner that installs neither).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from training import vocab  # noqa: E402
from training.am.dataset import collate  # noqa: E402
from training.am.model import AcousticModel  # noqa: E402
from training.features import mfcc  # noqa: E402


def _tiny_model(n_layers=1, n_hidden=32):
    return AcousticModel(n_hidden=n_hidden, n_layers=n_layers)


def test_model_forward_shape_and_normalization():
    model = _tiny_model()
    b, t = 3, 20
    feats = torch.randn(b, t, mfcc.FEATURE_DIM)
    lengths = torch.tensor([t, t - 5, t - 8])
    logp = model(feats, lengths)
    assert logp.shape == (b, t, vocab.VOCAB_SIZE)
    probs = logp.exp().sum(-1)  # log-softmax rows sum to 1 in prob space
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-4)


def test_feature_extract_is_123_dim():
    mel, deltas = mfcc.build_transforms()
    wav = torch.randn(1, mfcc.SAMPLE_RATE)  # 1 s of noise
    feats = mfcc.extract(wav, mel, deltas)
    assert feats.dim() == 2
    assert feats.size(1) == mfcc.FEATURE_DIM == 123


def test_cmvn_normalizes_to_zero_mean_unit_std():
    feats = [torch.randn(50, mfcc.FEATURE_DIM) * 3 + 7 for _ in range(4)]
    mean, std = mfcc.compute_cmvn(iter(feats))
    normed = torch.cat([mfcc.apply_cmvn(f, mean, std) for f in feats])
    assert torch.allclose(normed.mean(0), torch.zeros(mfcc.FEATURE_DIM), atol=1e-4)
    assert torch.allclose(normed.std(0, unbiased=False), torch.ones(mfcc.FEATURE_DIM), atol=1e-2)


def test_features_for_reads_flac(tmp_path):
    """The real FLAC path: soundfile load -> extract -> 123-dim, plus label encode."""
    import soundfile as sf

    from training.am.dataset import LibriSpeechFeatures, list_utterances

    d = tmp_path / "LibriSpeech" / "dev-clean" / "1" / "2"
    d.mkdir(parents=True)
    sf.write(str(d / "1-2-0000.flac"), (torch.randn(16000) * 0.1).numpy(), 16000, format="FLAC")
    (d / "1-2.trans.txt").write_text("1-2-0000 HELLO WORLD", encoding="utf-8")

    items = list_utterances(tmp_path / "LibriSpeech" / "dev-clean")
    assert len(items) == 1
    feats, labels = LibriSpeechFeatures(items)[0]
    assert feats.size(1) == 123
    assert labels.tolist() == vocab.encode("HELLO WORLD")


def test_collate_pads_and_packs_lengths():
    items = [
        (torch.randn(10, mfcc.FEATURE_DIM), torch.tensor(vocab.encode("HI"))),
        (torch.randn(7, mfcc.FEATURE_DIM), torch.tensor(vocab.encode("YES"))),
    ]
    feats, feat_len, labels, label_len = collate(items)
    assert feats.shape == (2, 10, mfcc.FEATURE_DIM)  # padded to the longest T
    assert feat_len.tolist() == [10, 7]
    assert label_len.tolist() == [2, 3]
    assert labels.numel() == 5  # concatenated 2 + 3


def test_overfit_one_batch_drives_loss_down():
    """Exercise the real loss/gradient/CTC wiring on synthetic data.

    Two short sequences a tiny LSTM should memorize, so CTC loss must fall sharply.
    """
    torch.manual_seed(0)
    model = _tiny_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ctc = torch.nn.CTCLoss(blank=vocab.BLANK_IDX, zero_infinity=True)

    t = 30
    feats = torch.randn(2, t, mfcc.FEATURE_DIM)
    feat_len = torch.tensor([t, t])
    targets = [torch.tensor(vocab.encode("CAT")), torch.tensor(vocab.encode("DOG"))]
    labels = torch.cat(targets)
    label_len = torch.tensor([3, 3])

    losses = []
    for _ in range(120):
        logp = model(feats, feat_len).transpose(0, 1)  # [T, B, N]
        loss = ctc(logp, labels, feat_len, label_len)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < 0.4 * losses[0], f"loss did not fall: {losses[0]:.2f} -> {losses[-1]:.2f}"
