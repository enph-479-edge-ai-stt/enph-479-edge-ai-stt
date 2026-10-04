# runtime/

The live demo program for the KV260's ARM cores (Ubuntu + PYNQ). Not started. Real-time processing is the term 2 goal (January to April); in term 1 the board only has to run the RNN on audio files fed in statically.

Its first job at startup, after loading the bitstream, is to stream the acoustic model's weight image (`am_fabric.mem` from a training run) into the fabric. The format and the load order are in `shared/specs/am_weight_image.md`.
