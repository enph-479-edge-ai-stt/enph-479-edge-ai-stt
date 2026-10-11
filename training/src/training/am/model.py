"""Acoustic model: stacked unidirectional LSTM -> linear -> log-softmax.

The network is the shared ``LstmNet``, 3 x 256 for the deployable small model. Input is
the 123-dim feature vector; output is the vocab size (CTC blank included), as per-frame
log-probabilities for ``nn.CTCLoss``.
"""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from training.features.fbank import FEATURE_DIM
from training.lstm.model import LstmNet
from training.vocab import VOCAB_SIZE


class AcousticModel(LstmNet):
    """Deep unidirectional LSTM acoustic model with a CTC output head."""

    x_frac = 5  # CMVN'd features reach about +-4, so the fabric gets them as int8 / 32

    def __init__(
        self,
        n_feats: int = FEATURE_DIM,
        n_hidden: int = 256,
        n_layers: int = 3,
        n_out: int = VOCAB_SIZE,
        dropout: float = 0.2,
    ) -> None:
        """Build the LSTM stack and the linear projection to ``n_out`` logits."""
        super().__init__(n_feats, n_hidden, n_layers, n_out, dropout)

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Map padded feature frames to per-frame log-probabilities.

        ``feats`` is ``[B, T, n_feats]`` padded with ``lengths`` ``[B]``; returns
        ``[B, T, n_out]``. Packing keeps the LSTM off the padding frames.
        """
        packed = pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.log_probs(out)
