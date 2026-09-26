# training/

The offline pipeline (Python). Today it trains the acoustic model on Colab.

- `notebooks/am_training.ipynb`: the Colab launcher (download, CMVN, train, save the weights and log to Drive).
- `src/training/data/librispeech.py`: LibriSpeech download (wget, md5 check, extract).
- `src/training/features/fbank.py`: 123-dim log-mel filterbank features and CMVN (provisional spec).
- `src/training/vocab.py`: 30-symbol character vocab (provisional).
- `src/training/am/`: the LSTM + CTC model, the dataset, and the training loop.

The later stages (char-LM, word-LM, decoding, QAT, export) haven't started.
