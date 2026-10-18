# training/

The offline pipeline (Python). Today it trains the acoustic model (AM) and the character model (CM) on Colab, fine-tunes each with 6-bit weights, and packs the result for the FPGA.

- `notebooks/am_training.ipynb`: the AM Colab launcher (download, CMVN, train, fine-tune with 6-bit weights, save both models, the logs and the FPGA weight image to Drive).
- `notebooks/cm_training.ipynb`: the CM Colab launcher (download, sample the LM text, train, fine-tune with 6-bit weights, save both models, the logs and the FPGA weight image to Drive).
- `src/training/data/librispeech.py`: LibriSpeech audio and LM-text download (wget, md5 check, extract). The archives are kept in a cache folder on Drive, `MyDrive/enph479-stt/data/`, so only the first session downloads from OpenSLR; later ones copy from Drive.
- `src/training/features/fbank.py`: 123-dim log-mel filterbank features and CMVN (provisional spec).
- `src/training/vocab.py`: 30-symbol character vocab (provisional).
- `src/training/lstm/`: everything the two models share. Both are a stack of LSTM layers with a linear output layer, so there is one of each of these:
  - `model.py`: the network (`LstmNet`), with the output layer both models end on.
  - `train.py`: the training loop (Adam, with the learning rate falling to 0 on one cosine over the run), including the fine-tune with the weights quantized (6 bits, on the FPGA's fixed-point grid; weights only).
  - `export.py`: packs a 6-bit model into the weight image the FPGA loads, `<name>_fabric.mem` (the weight memory's contents) and `<name>_fabric.json` (dimensions and input scale for the ARM). The layout is specified in `shared/specs/weight_image.md`. Also `save_run`, the one definition of what a run leaves on Drive: both checkpoints, both logs and the image.
- `src/training/am/`: what is the AM's own. Feature input with padding, the dataset and `load_data` (download, CMVN), the CTC loss, and the test loss and greedy CER logged after every epoch.
- `src/training/cm/`: what is the CM's own (the paper's char-LM). One-hot input with carried state, the EOS-joined text stream and `load_data` (download, sample the corpus), truncated-BPTT batching, bits per character and sampling.
- `src/training/wlm/model.py`: the word LM, OpenSLR's pruned trigram over the same LM text (downloaded by `data/librispeech.py`), read with KenLM and adapted to score one word at a time; also word perplexity and OOV rate on a test set. Needs the `wlm` extra (KenLM, built from source: `uv sync --extra wlm`, with `CC=clang CXX=clang++` under a uv-managed Python; not on Windows, where its tests skip).
- `src/training/decode/`: the decoder. `beam.py` is CTC prefix beam search fused with the CM and the WLM (the paper's N-best search), plain Python so it runs anywhere; `runtime/` has a line-for-line copy. `golden.py` writes the golden case both copies must reproduce (`shared/golden/decode/`). `run.py` is what the decode notebook uses: loading the checkpoints, AM log-probs over a test set, the CM in torch behind the search's interface, WER/CER and the grid search over the LM weights.
- `notebooks/decode.ipynb`: the decode Colab launcher (load an AM and a CM run from Drive, decode dev-clean, tune the LM weights, report greedy / beam / + CM / + WLM / + both, save the results to Drive).

A new LSTM model goes the same way: subclass `LstmNet`, hand `lstm.train.fit` its losses and its evaluation, and `export` works on it as is.

The later stages (activation and cell quantization, the integer reference model) haven't started.
