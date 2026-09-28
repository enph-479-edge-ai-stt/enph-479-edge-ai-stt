# enph-479-edge-ai-stt

[![CI](https://github.com/enph-479-edge-ai-stt/enph-479-edge-ai-stt/actions/workflows/ci.yml/badge.svg)](https://github.com/enph-479-edge-ai-stt/enph-479-edge-ai-stt/actions/workflows/ci.yml)

ENPH 479 capstone (UBC Engineering Physics, 2026W). An FPGA speech-recognition system reproducing "FPGA-Based Low-Power Speech Recognition with Recurrent Neural Networks" (Lee et al., arXiv:1610.00552): a deep LSTM acoustic model trained with CTC, a character-level LSTM language model, and a trigram word language model, fused by an N-best beam search, running quantized RNN inference on a Kria KV260.

Team: Kai Asaoka, Sudharshan Kannan, Andrew Du, Anubhav Saini.

## What it does

A person speaks into a microphone and the KV260 turns the audio into text in real time. A browser page shows the live transcript along with latency and power. Everything between the mic and the browser runs on the board: feature extraction and beam search on the ARM cores, the 6-bit-quantized LSTM acoustic model (and character LM) on the FPGA fabric. The goal is to match the system the paper actually ran on its FPGA: the small model (3x256 acoustic LSTM, 2x256 char-LM) with 6-bit weights, which scored 14.02% word error rate at 9.24 W and 4.12x real time. The paper's 8.79% headline comes from a larger floating-point model on a GPU that doesn't fit this board.

## How it fits together

An offline pipeline trains the models on a GPU and compiles them into an FPGA bitstream; the on-board runtime then runs that bitstream live, streaming audio in and text out.

```
offline (PC + GPU)
  audio -> features -> LSTM + CTC training -> quantization (QAT) -> QONNX export -> bitstream (Vivado)
  text  -> char-LM (LSTM) and word-LM (KenLM trigram)

on board (KV260)
  mic -> features -> [fabric: acoustic model + char-LM] -> beam search + word-LM -> live transcript
```

## Roadmap

The live, real-time system above is the end goal. We get there in two stages:

- **Term 1 (September to January): the RNN running on the fabric, offline.** The hardware runs the trained LSTM on our test data, fed in statically from audio files. In effect it's `model.score()` on audio files, with the FPGA fabric doing the RNN math instead of the GPU. No live audio yet.
- **Term 2 (January to April): real-time processing.** Streaming audio from the microphone through the same model, and building out the live pipeline described in "What it does".

## Repository layout

- `hardware/`: Verilog RTL for the LSTM accelerator. In progress; several modules are still empty stubs.
- `training/`: the offline pipeline (Python). Today: LibriSpeech download, 123-dim filterbank features, and the LSTM + CTC acoustic model, trained from `training/notebooks/am_training.ipynb` on Colab.
- `runtime/`: the on-board program for the KV260. Not started; the real-time part is the term 2 goal.
- `shared/`: frozen cross-vertical specs and golden vectors. Nothing frozen yet.

## Getting the code

```
git clone https://github.com/enph-479-edge-ai-stt/enph-479-edge-ai-stt.git
```

## Development

Each vertical is its own `uv` project (own `pyproject.toml`, lockfile, and Python version) — they run on different machines. Work inside the one you're touching:

```
cd training            # the offline pipeline (Python pinned to 3.12.0)
uv sync                # create the venv and install dev tooling
uv run ruff check .    # lint
uv run pytest          # tests
```

Lint and tests must pass before opening a PR; CI enforces both. See `AGENTS.md` for the full workflow.

## Reference

Lee, Hwang, Park, Choi, Shin, and Sung, "FPGA-Based Low-Power Speech Recognition with Recurrent Neural Networks," arXiv:1610.00552.
