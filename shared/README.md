# shared/

Cross-vertical specs, golden vectors, and the code more than one vertical runs.

- `specs/weight_image.md`: how a model's weights are laid out for the fabric (fixed-point formats, memory map, stream order), for the acoustic model and the character model. Provisional until `hardware/` builds to it.

- `src/shared/`: the `stt-shared` Python package, which training depends on and the runtime will (`../shared` as a path dependency). Python 3.10 or newer (the board's), numpy only, plus KenLM for the word LM (the `wlm` extra).
  - `vocab.py`: the 30-symbol character vocab (provisional). Changing it invalidates every trained checkpoint.
  - `decode/beam.py`: CTC prefix beam search fused with the char-LM and the word LM (the paper's N-best search), plain Python. Training runs it with the models in torch, the board with them on the fabric.
  - `decode/wlm.py`: OpenSLR's pruned trigram through KenLM, a word at a time, and word perplexity on a test set.
  - `decode/golden.py`: writes `golden/decode/`.

- `golden/decode/`: the beam search's golden case, which pins its output. See its README.

Still living in the training code, provisional: the feature spec (`training/src/training/features/fbank.py`).

`uv sync --extra wlm` and `uv run pytest` here, as in `training/`. kenlm builds from source, so not on Windows, where its tests skip.
