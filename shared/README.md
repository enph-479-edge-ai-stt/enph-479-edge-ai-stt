# shared/

Cross-vertical specs, golden vectors, and the code more than one vertical runs.

- `specs/weight_image.md`: how a model's weights are laid out for the fabric (fixed-point formats, memory map, stream order), for the acoustic model and the character model. Provisional until `hardware/` builds to it.

- `src/shared/`: the `stt-shared` Python package, which training depends on and the runtime will (`../shared` as a path dependency). Python 3.10 or newer (the board's), numpy only, plus KenLM for the word LM (the `wlm` extra).
  - `vocab.py`: the 30-symbol character vocab (provisional). Changing it invalidates every trained checkpoint.
  - `decode/beam.py`: CTC prefix beam search fused with the char-LM and the word LM (the paper's N-best search), plain Python. The readable reference: it writes the golden case, and the C++ search below is a port of it.
  - `decode/beam_cpp*.so`: the C++ search, built from `cpp/` (not in git). The one that runs: `decode.ipynb` decodes with it, with the models in torch, and the board will with them on the fabric. About 7x faster than `beam.py` on the board's A53s (0.14x against 0.96x real time at beam 128), with the same N-best and scores.
  - `decode/wlm.py`: OpenSLR's trigram through KenLM, a word at a time, and word perplexity on a test set.
  - `decode/golden.py`: writes `golden/decode/`.

- `cpp/`: `beam_cpp.cpp`, the port of `decode/beam.py` and `decode/wlm.py` with KenLM's query code compiled in (a pybind11 module, same `beam_search(logp, cfg, cm, wlm)` call; its `WordLM(arpa)` stands in for `KenLM`), and `build.sh`, which builds it into `src/shared/decode/`. Needs g++, zlib, pybind11 and the kenlm-0.3.0 sdist: `PYTHON=<python> cpp/build.sh <kenlm source dir>`, about 2 minutes on the board, less on a PC. It keeps `beam.py`'s order of floating-point operations, so on one machine the two return exactly the same output.

- `golden/decode/`: the beam search's golden case, which pins its output. See its README.

Still living in the training code, provisional: the feature spec (`training/src/training/features/fbank.py`).

`uv sync --extra wlm` and `uv run pytest` here, as in `training/`. kenlm builds from source, so not on Windows, where its tests skip. The C++ search's tests (`tests/test_beam_cpp.py`) skip until it is built: `uv run --with pybind11 bash cpp/build.sh <kenlm source dir>`, as CI does.
