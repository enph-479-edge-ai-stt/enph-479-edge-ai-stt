"""The network both models are: stacked unidirectional LSTM -> linear output layer.

Fixed by the hardware: unidirectional (streaming inference can't see the future), no
peepholes (``nn.LSTM`` has none), 256 units per layer for the deployable small models.
The acoustic and character models subclass it with their own input handling. The
training loop, the 6-bit fine-tune and the weight image export work on this class.
"""

from __future__ import annotations

import torch
from torch import nn


class LstmNet(nn.Module):
    """LSTM stack ``lstm`` and output layer ``proj``."""

    x_frac: int  # the fabric gets a layer-0 input as an integer at scale 2**-x_frac

    def __init__(
        self, n_in: int, n_hidden: int, n_layers: int, n_out: int, dropout: float = 0.0
    ) -> None:
        """Build the LSTM stack and the linear projection to ``n_out`` logits."""
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_in,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,  # nn.LSTM warns on 1 layer
        )
        self.proj = nn.Linear(n_hidden, n_out)

    def log_probs(self, out: torch.Tensor) -> torch.Tensor:
        """LSTM outputs ``[..., n_hidden]`` -> log-probabilities over the ``n_out`` symbols."""
        return self.proj(out).log_softmax(dim=-1)
