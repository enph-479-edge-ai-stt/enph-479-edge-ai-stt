"""Character model (CM): one-hot character -> stacked unidirectional LSTM -> linear -> log-softmax.

As in the paper: the input is the current character one-hot encoded and the output is
the distribution over the next one. The network is the shared ``LstmNet``, 2 x 256 for
the deployable small model. It shares the AM's 30-symbol vocab so their indices line up
in beam search; the CTC blank never occurs in text, so the LM learns to give it ~zero
probability. The LSTM state is returned so the caller can carry it on: across BPTT
chunks in training, and per beam hypothesis in decoding.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from training.lstm.model import LstmNet
from training.vocab import VOCAB_SIZE

State = tuple[torch.Tensor, torch.Tensor]


class CharModel(LstmNet):
    """Deep unidirectional LSTM character language model."""

    x_frac = 0  # one-hot: the fabric gets the current character's input as the integer 1

    def __init__(self, n_vocab: int = VOCAB_SIZE, n_hidden: int = 256, n_layers: int = 2) -> None:
        """Build the LSTM stack (one-hot input) and the linear projection to next-char logits."""
        super().__init__(n_vocab, n_hidden, n_layers, n_vocab)

    def forward(
        self, chars: torch.Tensor, state: State | None = None
    ) -> tuple[torch.Tensor, State]:
        """Map characters ``[B, T]`` (int) to next-character log-probs ``[B, T, n_vocab]``.

        ``state`` is the ``(h, c)`` the LSTM starts from (zeros if ``None``); the
        state after the last step is returned with the log-probs.
        """
        x = F.one_hot(chars.long(), self.lstm.input_size).float()
        out, state = self.lstm(x, state)
        return self.proj(out).log_softmax(dim=-1), state
