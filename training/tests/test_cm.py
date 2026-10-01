"""Character-model tests: corpus sampling and encoding, model, BPTT chunking, training loop.

Synthetic text only (a tiny gzipped corpus in a temp dir), CPU, no network. Skips
when torch is absent, which is the case on CI (it isn't a locked dep).
"""

from __future__ import annotations

import gzip
import math

import pytest

torch = pytest.importorskip("torch")

from training import vocab  # noqa: E402
from training.cm.dataset import sample_sentences, to_stream  # noqa: E402
from training.cm.model import CharModel  # noqa: E402
from training.cm.train import (  # noqa: E402
    TrainConfig,
    _chunks,
    _streams,
    bits_per_char,
    train,
)

SENTENCES = ["HELLO WORLD", "THE CAT SAT", "IT'S FINE", "A DOG RAN HOME"]


def _write_corpus(path, sentences):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("".join(s + "\n" for s in sentences))
    return path


def test_sample_keeps_all_or_nothing_and_shuffles(tmp_path):
    gz = _write_corpus(tmp_path / "corpus.txt.gz", [f"SENTENCE {c}" for c in "ABCDEFGHIJ"])
    kept = sample_sentences(gz, keep=1.0)
    assert sorted(kept) == [f"SENTENCE {c}" for c in "ABCDEFGHIJ"]
    assert kept != sorted(kept)  # shuffled out of corpus order (seed 0)
    assert sample_sentences(gz, keep=0.0) == []


def test_sample_fraction_is_roughly_keep(tmp_path):
    gz = _write_corpus(tmp_path / "corpus.txt.gz", ["A B"] * 2000)
    assert 400 < len(sample_sentences(gz, keep=0.25)) < 600


def test_to_stream_joins_with_eos_and_matches_vocab_encode():
    stream = to_stream(["HI", "IT'S"])
    eos = vocab.EOS_IDX
    assert stream.tolist() == [eos, *vocab.encode("HI"), eos, *vocab.encode("IT'S"), eos]


def test_to_stream_rejects_out_of_vocab():
    with pytest.raises(ValueError, match="out-of-vocab"):
        to_stream(["hello"])


def test_streams_and_chunks_pair_inputs_with_next_chars():
    stream = to_stream(SENTENCES)
    streams = _streams(stream, batch_size=3)
    assert streams.shape == (3, len(stream) // 3)
    pieces = list(_chunks(streams, bptt=5))
    xs = torch.cat([x for x, _ in pieces], dim=1)
    ys = torch.cat([y for _, y in pieces], dim=1)
    assert torch.equal(xs, streams[:, :-1])  # every position but the last is an input
    assert torch.equal(ys, streams[:, 1:])  # ... and its target is the next character


def test_model_outputs_log_probs_and_carries_state():
    model = CharModel(n_hidden=16, n_layers=2)
    chars = torch.randint(1, vocab.VOCAB_SIZE, (3, 7))
    logp, (h, c) = model(chars)
    assert logp.shape == (3, 7, vocab.VOCAB_SIZE)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(3, 7), atol=1e-4)
    assert h.shape == c.shape == (2, 3, 16)
    # Running in two halves with the state carried equals one pass.
    a, state = model(chars[:, :4])
    b, _ = model(chars[:, 4:], state)
    assert torch.allclose(torch.cat([a, b], 1), logp, atol=1e-5)


def test_untrained_model_is_near_uniform_bpc():
    torch.manual_seed(0)
    streams = _streams(to_stream(SENTENCES * 5), batch_size=2)
    bpc = bits_per_char(CharModel(n_hidden=16, n_layers=1), streams, bptt=8)
    assert abs(bpc - math.log2(vocab.VOCAB_SIZE)) < 0.5


def test_overfit_one_stream_drives_loss_down():
    """The real loss/gradient wiring on text a tiny LSTM can memorize."""
    torch.manual_seed(0)
    model = CharModel(n_hidden=32, n_layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    stream = torch.from_numpy(to_stream(SENTENCES)).unsqueeze(0)
    x, y = stream[:, :-1], stream[:, 1:]
    losses = []
    for _ in range(150):
        logp, _ = model(x)
        loss = torch.nn.functional.nll_loss(logp.flatten(0, 1), y.long().flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.2 * losses[0]


def test_train_runs_end_to_end_on_tiny_data(tmp_path):
    streams = {"train": to_stream(SENTENCES * 10), "test": to_stream(SENTENCES)}
    cfg = TrainConfig(n_hidden=16, n_layers=2, batch_size=2, bptt=10, epochs=1, device="cpu")
    model = train(cfg, streams, tmp_path / "train.log")
    assert isinstance(model, CharModel)
    log = (tmp_path / "train.log").read_text(encoding="utf-8")
    assert "[epoch 0]" in log
    assert "sample:" in log
