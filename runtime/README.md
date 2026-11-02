# runtime/

The live demo program for the KV260's ARM cores (Ubuntu + PYNQ). Not started. Real-time processing is the term 2 goal (January to April); in term 1 the board only has to run the RNN on audio files fed in statically.

Its first job at startup, after loading the bitstream, is to stream the models' weight images (`am_fabric.mem`, and `cm_fabric.mem` once the char-LM is on the fabric, from the training runs) into the fabric. The format and the load order are in `shared/specs/weight_image.md`.

The decoder it runs on the fabric's output (beam search with the char-LM and the word LM) and the vocab are in `shared/src/shared/` (the `stt-shared` package), not here: training runs the same code on Colab. The runtime takes it as a path dependency on `../shared` once it has a `pyproject.toml`.
