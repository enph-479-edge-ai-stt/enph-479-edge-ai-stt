# enph-479-edge-ai-stt

[![CI](https://github.com/enph-479-edge-ai-stt/enph-479-edge-ai-stt/actions/workflows/ci.yml/badge.svg)](https://github.com/enph-479-edge-ai-stt/enph-479-edge-ai-stt/actions/workflows/ci.yml)

ENPH 479 capstone (UBC Engineering Physics, 2026W). An FPGA speech-recognition system reproducing "FPGA-Based Low-Power Speech Recognition with Recurrent Neural Networks" (Lee et al., arXiv:1610.00552): a deep LSTM acoustic model trained with CTC, a character-level LSTM language model, and a trigram word language model, fused by an N-best beam search, running quantized RNN inference on a Kria KV260.

Team: Kai Asaoka, Sudharshan Kannan, Andrew Du, Anubhav Saini.

## What it does

A person speaks into a microphone and the KV260 turns the audio into text in real time. A browser page shows the live transcript along with latency and power. Everything between the mic and the browser runs on the board: feature extraction and beam search on the ARM cores, the 6-bit-quantized LSTM acoustic model (and character LM) on the FPGA fabric. The goal is to match the reference paper's result, about 8.79% word error rate at roughly 10 W, faster than real time.

## How it fits together

An offline pipeline trains the models on a GPU and compiles them into an FPGA bitstream; the on-board runtime then runs that bitstream live, streaming audio in and text out.

```
offline (PC + GPU)
  audio -> features -> LSTM + CTC training -> quantization (QAT) -> QONNX export -> bitstream (Vivado)
  text  -> char-LM (LSTM) and word-LM (KenLM trigram)

on board (KV260)
  mic -> features -> [fabric: acoustic model + char-LM] -> beam search + word-LM -> live transcript
```

## Repository layout

- `hardware/`: the FPGA design. Verilog RTL for the LSTM accelerator (PE array, LUT-based sigmoid/tanh, weight BRAM, FSM control) plus testbenches.
- `training/`: the offline pipeline (Python). Data prep, feature extraction, acoustic-model and language-model training, quantization-aware training, and export to QONNX.
- `runtime/`: the on-board program that runs the live demo on the KV260's ARM cores under PYNQ Linux.
- `shared/`: specifications and reference vectors used across the project (feature format, vocabulary, quantization scheme, bit-exact test vectors).

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
