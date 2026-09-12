# shared/

Specifications and reference data that every part of the project reads from, kept in one place so the training, FPGA, and runtime paths stay in agreement.

Holds:

- the feature format (16 kHz audio, 25 ms / 10 ms framing, 40 log-mel + energy + delta + double-delta = 123 dimensions, CMVN)
- the 31-symbol vocabulary (26 letters, 3 punctuation, end-of-sentence, CTC blank)
- the quantization scheme (6-bit weights, 8-bit activations, 16-bit LSTM cell)
- bit-exact reference vectors: audio-to-features golden outputs, and the QONNX CPU-execution outputs the board must match
