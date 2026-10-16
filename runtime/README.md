# runtime/

The live demo program for the KV260's ARM cores (Ubuntu + PYNQ). Real-time processing is the term 2 goal (January to April); in term 1 the board only has to run the RNN on audio files fed in statically.

What exists: the decoder.

- `src/runtime/decode/beam.py`: CTC prefix beam search fused with the char-LM and the word LM, the paper's N-best search. A line-for-line copy of training's (`training/src/training/decode/beam.py`), which is the reference; both must reproduce `shared/golden/decode/`. The char-LM comes in through a small interface, so on the board it will be the fabric (one context per beam entry), and in tests a bigram table.
- `src/runtime/decode/wlm.py`: the word LM, OpenSLR's pruned LibriSpeech trigram read with KenLM (`3-gram.pruned.1e-7.arpa.gz`, 34 MB, loads in about a second).
- `src/runtime/vocab.py`: a copy of training's vocab, checked against the golden case.

Python 3.10, the board's. Dev loop as in `training/`: `uv sync --extra wlm` (the extra is KenLM, which builds from source: with a uv-managed Python set `CC=clang CXX=clang++`; it doesn't build on Windows, where plain `uv sync` works and the KenLM tests skip), `uv run ruff check .`, `uv run pytest`.

The search is plain Python. On a desktop it decodes a noisy synthetic 9 s utterance at beam 128 in about 3 s; the A53s are several times slower, so it will need speeding up (or a smaller beam) for the live demo.

Its first job at startup, after loading the bitstream, is to stream the models' weight images (`am_fabric.mem`, and `cm_fabric.mem` once the char-LM is on the fabric, from the training runs) into the fabric. The format and the load order are in `shared/specs/weight_image.md`.
