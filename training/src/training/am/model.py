"""Acoustic model: stacked unidirectional LSTM -> linear -> log-softmax.

Fixed by the hardware (see ``acoustic-model-spec.md``): unidirectional (streaming
inference can't see the future), 3 x 256 for the deployable small model, no
peepholes (``nn.LSTM`` has none, and the paper's peepholes are dropped by
decision). Dims are constructor parameters, not literals, so the smoke test can
shrink to e.g. 2 x 128 on CPU. Input is the 123-dim feature vector; output is the
vocab size (CTC blank included), as per-frame log-probabilities for ``nn.CTCLoss``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from training.features.mfcc import FEATURE_DIM
from training.vocab import VOCAB_SIZE


class AcousticModel(nn.Module):
    def __init__(
        self,
        n_feats: int = FEATURE_DIM,
        n_hidden: int = 256,
        n_layers: int = 3,
        n_out: int = VOCAB_SIZE,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feats,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if n_layers > 1 else 0.0,  # nn.LSTM warns on 1 layer
        )
        self.proj = nn.Linear(n_hidden, n_out)

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """feats ``[B, T, n_feats]`` padded, lengths ``[B]`` -> log-probs
        ``[B, T, n_out]``. Packing keeps the LSTM off the padding frames."""
        packed = pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.proj(out).log_softmax(dim=-1)
