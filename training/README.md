# training/

The offline pipeline (Python). Today it trains the acoustic model (AM) and the character model (CM) on Colab.

- `notebooks/am_training.ipynb`: the AM Colab launcher (download, CMVN, train, save the weights and log to Drive).
- `notebooks/cm_training.ipynb`: the CM Colab launcher (download, sample the LM text, train, save to Drive).
- `src/training/data/librispeech.py`: LibriSpeech audio and LM-text download (wget, md5 check, extract).
- `src/training/features/fbank.py`: 123-dim log-mel filterbank features and CMVN (provisional spec).
- `src/training/vocab.py`: 30-symbol character vocab (provisional).
- `src/training/am/`: the LSTM + CTC model, the dataset, and the training loop.
- `src/training/cm/`: the character model, the paper's char-LM (one-hot in, 2x256 LSTM), the EOS-joined text stream, and the truncated-BPTT training loop.

The later stages (word-LM, decoding, QAT, export) haven't started.
