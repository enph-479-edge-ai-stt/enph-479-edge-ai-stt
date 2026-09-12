# training/

The offline pipeline (Python, runs on a GPU). It produces the QONNX file and frozen quantized weights that the `hardware/` flow synthesizes into a bitstream.

## Stages

1. `data`: LibriSpeech download, transcript cleaning, 31-symbol vocab build
2. `features`: 123-dim log-mel + delta + double-delta + CMVN, cached. This is the reference the runtime feature code must match.
3. `am`: LSTM acoustic model trained with CTC
4. `charlm`: character-level LSTM language model
5. `wordlm`: KenLM trigram word language model (ARPA to binary)
6. `decode`: N-best beam search fusing acoustic, char-LM, and word-LM scores; WER via jiwer
7. `qat`: Brevitas quantization-aware training (6-bit weights, 8-bit activations, 16-bit cell)
8. `export`: Brevitas to QONNX, with a bit-true check against the fixed-point reference
