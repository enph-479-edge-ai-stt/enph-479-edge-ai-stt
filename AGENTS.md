# AGENTS.md

Software monorepo for an FPGA speech-recognition system, ENPH 479 capstone (UBC Engineering Physics, 2026W). Team: Kai Asaoka, Sudharshan Kannan, Andrew Du.

Reproduces Lee et al., "FPGA-Based Low-Power Speech Recognition with Recurrent Neural Networks" (arXiv:1610.00552) on a modern board: a deep LSTM acoustic model trained with CTC, a character-level LSTM language model, and a KenLM trigram word LM, fused by N-best beam search. Quantized RNN inference runs on the FPGA fabric of a Kria KV260; everything else runs on the board's ARM cores under Linux.

## Three verticals

The repo is split by the three things that get built, each with its own toolchain and its own machine. The verticals share source and contracts, not a build environment.

| Vertical | Folder | Runs on | Produces |
|---|---|---|---|
| 1. Training data + ML | `training/` | Google Colab (GPU) / any PC | QONNX file + frozen quantized weights |
| 2. Hardware + flashing to FPGA | `fpga/` | Local Linux box with Vivado / Vitis HLS | Bitstream (`.bit` + `.hwh`) |
| 3. Runtime on the SoM Linux | `runtime/` | KV260 ARM cores, Ubuntu + PYNQ | The live mic-to-browser demo |

`shared/` holds the cross-vertical contracts (feature spec, vocab, quant scheme, golden vectors). Nothing in `shared/` belongs to one vertical.

```
offline (PC + GPU)
  audio -> features -> LSTM+CTC training -> QAT (Brevitas) -> QONNX export -+
  text  -> char-LM (LSTM)                                                  +-> HLS / Vivado -> bitstream
  text  -> word-LM (KenLM trigram) ----------------------------------------+

on board (KV260)
  USB mic -> numpy features -> [fabric: AM + char-LM] -> beam search + word-LM -> websocket -> browser
```

### 1. `training/` (data + ML, Python)

Python package. Notebooks in `training/notebooks/` are thin launchers only; real code lives in modules so Colab sessions are disposable (clone, run, die).

Stages, in order: `data` (LibriSpeech download, transcript cleanup, 31-symbol vocab) -> `features` (123-dim, cached) -> `am` (LSTM + CTC) -> `charlm` -> `wordlm` (KenLM, no training) -> `decode` (beam search prototype, WER via jiwer) -> `qat` (Brevitas QuantLSTM) -> `export` (QONNX + bit-true check).

- Data: LibriSpeech from OpenSLR 12, LM text and prebuilt ARPAs from OpenSLR 11. Train on `train-clean-100` first, `dev-clean` for tuning, `test-clean` touched once.
- Colab rules: code in GitHub, artifacts on Drive, shards copied to `/content` scratch before training, resume-from-checkpoint is the default path, pin versions. Clone without the submodule on Colab.
- Stack: PyTorch, torchaudio transforms (TorchCodec for file I/O), Brevitas, qonnx, KenLM, jiwer, pyctcdecode as the reference decoder only.
- The feature pipeline here is the reference the runtime numpy code must bit-match.

### 2. `fpga/` (hardware design + build + flash)

Turns the trained model into a bitstream and gets it onto the board.

- RTL lives in the git submodule `fpga/enph-479-stt-neural-network/` (Andrew's repo, `AndrewD0/enph-479-stt-neural-network`). Hand-written Verilog modelled on the paper's LSTM tile: `pe_unit` (8b x 6b MAC, 24b accumulator), `pe_array`, `pe_buffer` (i/f/o/c gate results), `lstm_epu`, `weight_bram` (packed rows, 6b weights), `sigmoid_lut` / `tanh_lut`, `fsm_controller`, `output_tile`, `sr_accelerator_top`. Several files are still empty stubs. Testbenches in `hardware/testbenches/`.
- The original plan was FINN-GL (Brevitas -> QONNX Scan -> FINN-GLSTM-Hw HLS templates -> Vitis HLS -> Vivado). The submodule is a hand-RTL path instead. Both are legitimate; which one is primary is an open team decision. Either way the bitstream must match the QONNX CPU execution bit-for-bit (gate V3 below).
- Build machine: Vivado + Vitis HLS on Linux, ~32 GB RAM, 100+ GB disk. Never on Colab, never on the board. Build outputs (`.bit`, `.xsa`, `.jou`, `.log`, `.Xil/`) are gitignored.
- Weight loading: either baked into the bitstream as BRAM init, or written over AXI at boot. AXI boot-write is preferred for iteration speed (swap weights without a rebuild). Undecided.
- Submodule workflow: `git submodule update --init --recursive` after clone. To pull new hardware work: `git -C fpga/enph-479-stt-neural-network pull`, then commit the pointer bump here.

### 3. `runtime/` (SoM Linux program)

The live demo on the KV260 ARM cores under Ubuntu + Kria-PYNQ. Planned modules (create folders as each starts): `capture/` (ALSA + sounddevice ring buffer), `features_np/` (numpy port of the training features, must bit-match `shared/golden/`), `pynq_driver/` (Overlay / allocate / DMA / MMIO wrapper), `decoder/` (beam search + kenlm, Python prototype then C port), `frontend/` (websocket + browser dashboard), `bare_metal/` (stretch, no-OS C variant for the power delta).

Deliberately absent on the board: PyTorch, pyctcdecode, any RNN math in software. The ARM only sees feature frames going in and 31-dim probability vectors coming out.

## Hardware (confirmed)

Board and chip facts below were checked against the AMD K26 product brief and the reference paper PDF on 2026-09-12. DSP/LUT/BRAM/URAM counts come from the DS987 datasheet (checked 2026-07-03).

**Target: Kria KV260 starter kit** (~$284). Three nested parts: KV260 carrier board -> K26 SOM -> XCK26 chip (Zynq UltraScale+ MPSoC, 16 nm).

| Resource | Spec | Our use |
|---|---|---|
| PS: application CPU | Quad-core 64-bit Arm Cortex-A53 (up to 1.33 GHz) | Linux, features, beam search, web server |
| PS: real-time CPU | Dual-core Arm Cortex-R5F | Unused; bare-metal stretch landing spot |
| PS: GPU | Mali-400 MP2 | Irrelevant |
| PL: logic | 256K system logic cells (~117K LUT / ~234K FF) | State machines, DMA plumbing, activations |
| PL: DSP slices | 1,248 | MAC array (paper used 512 PEs) |
| PL: on-chip SRAM | 26.6 Mb (144 BRAM + 64 URAM blocks) | All weights resident on-chip (~8.8 Mb for the paper's large model) |
| Memory | 4 GB 64-bit DDR4, 16 GB eMMC, QSPI boot flash | Word-LM ARPA lives in DDR4 |
| Carrier I/O | 4x USB 2.0/3.0, 1 GbE, DisplayPort/HDMI, microSD, 12 V jack | USB mic (no analog audio in), ethernet to the browser |

**Flash and boot path.** Boot firmware (FSBL, PMU firmware, U-Boot) lives in QSPI on the SOM itself. The microSD carries Ubuntu 22.04 for Kria; Kria-PYNQ is installed on top (`sudo bash install.sh -b KV260`, JupyterLab on port 9090). The fabric is programmed from Linux, not over JTAG: `pynq.Overlay("design.bit")` calls the kernel FPGA Manager. PYNQ needs the `.bit` and the matching `.hwh` together, always scp both. AMD's `xmutil` is the alternative loader.

**Day-to-day loop:** build on the Linux box -> scp `.bit` + `.hwh` to the board -> ssh or Jupyter -> `Overlay()` -> stream frames over AXI-DMA -> read probabilities -> decode on ARM.

**Reference paper hardware, for comparison** (from the PDF): Xilinx XC7Z045 on a ZC706, 2.18 MB on-chip memory, 512 PEs (two arrays of 256) at 100 MHz, ARM at 800 MHz running the N-best search. 6-bit weights, 8-bit signals, 16-bit LSTM cells. Peephole LSTM. The FPGA ran the small model (3x256 AM, 2x256 char-LM, beam 128) at 9.24 W and 4.12x real time. The large model (4x512 AM, 2x512 char-LM) gave the headline 8.79% WER / 3.90% CER on WSJ eval92 and needs an UltraScale-class part, which the KV260 is. Power will not be apples-to-apples (28 nm vs 16 nm); frame it as a reproduction on a current platform.

## Cross-vertical contracts (why this is one repo)

The verification gates are bit-exact contracts that cross folder boundaries. Keep them in `shared/` and treat any change there as a breaking change for all three verticals.

- **Features:** 16 kHz mono, 25 ms Hamming window, 10 ms hop, 40 log-mel + energy + delta + double-delta = 123 dims, normalized on training-set statistics, quantized to int8 at the fabric boundary. 100 frames/s. Pin the exact CMVN recipe in `shared/specs/` before anything downstream bakes it in.
- **Vocabulary:** 31 symbols (26 letters, 3 punctuation, end-of-sentence, CTC blank). Fixed integer mapping; changing it invalidates every trained artifact.
- **Quantization:** 6-bit weights, 8-bit activations, 16-bit cell state.
- **Fabric interface:** 123 x int8 in per frame, 31 x int8 out per frame, over AXI-DMA. Char-LM control (char + context id) over MMIO.
- **Golden vectors:** audio -> features, and QONNX CPU-execution outputs. The board must match the QONNX CPU run bit-for-bit.

Gates: V1 greedy CER during float training; V2 WER holds after QAT; V3 board == QONNX CPU, bit-for-bit; V4 end-to-end WER through the board; V5 sustained real-time factor >= 1 and mic-to-screen latency in budget. No power or latency number is reported until V3 passes.

## Git workflow

- **Always create worktrees from `main`.** Never from another feature branch, and never from whatever branch happens to be checked out. Fetch first so `main` is current:
  `git fetch origin && git worktree add ../stt-<name> -b feat/<name> origin/main`
- `main` is the integration branch. Rebase or merge `main` in before opening a PR.
- Before opening a PR, lint and tests must pass in the affected vertical (e.g. `cd training`): `uv run ruff check .` and `uv run pytest` (with `UV_PROJECT_ENVIRONMENT` set — see Environments). CI (`.github/workflows/ci.yml`) runs them per vertical on every PR to `main` and on pushes to `main`.
- Branch names: `feat/<thing>`, `fix/<thing>`. One vertical per branch where possible.
- Commit messages follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/#specification): `<type>(<scope>): <description>`, imperative, lowercase, no trailing period. Types: `feat`, `fix`, `docs`, `refactor`, `test`, `build`, `ci`, `chore`. Scope is the vertical or shared area it touches: `training`, `fpga`, `runtime`, `shared`; omit it for repo-wide changes. Anything that changes a contract in `shared/` is a breaking change: add `!` after the scope and a `BREAKING CHANGE:` footer saying which artifacts it invalidates.
- Never commit data, checkpoints, ARPA files, bitstreams, or venvs. `.gitignore` covers them; if you add a new artifact type, add it there.
- Submodule pointer bumps are their own commit (`chore(fpga): bump hw submodule`).
- Do not commit or push unless asked.

## Environments

- Each vertical is its own **independent uv project** — its own `pyproject.toml`, `.python-version`, `uv.lock`, and venv — because the verticals run on different machines, hardware, and even Python versions. This is deliberately *not* a uv workspace (a workspace shares one lockfile / venv / Python version across members, which is the opposite of what we want). Nothing builds all three at once.
- `training/` is the only Python project today: src layout (package under `training/src/training/`, `import training.data...`), pinned to Python 3.12.0, hatchling backend, with ruff + pytest config and a `dev` dependency group (ruff, pytest) in `training/pyproject.toml`. `runtime/` gets the same treatment when its first module lands; `hardware/` is Verilog, no Python. The ML stack (torch, brevitas, qonnx, kenlm, jiwer, ...) is intentionally kept out of `dependencies` until each stage lands, so `uv sync` stays fast.
- Venvs must live outside OneDrive (OneDrive evicts package files and corrupts in-folder venvs). This repo is inside OneDrive, so before running any uv command point uv at an external venv by setting `UV_PROJECT_ENVIRONMENT` to an absolute path outside OneDrive (one per vertical) — uv's default in-folder `./.venv` must not be used. Building from the OneDrive-hosted source is fine; only the installed venv needs to sit elsewhere.
- Dev loop (run inside the vertical, e.g. `cd training`, with `UV_PROJECT_ENVIRONMENT` set): `uv sync` creates/updates the venv and installs the `dev` group; `uv run ruff check .` lints (notebooks included), `uv run ruff format` formats, `uv run pytest` runs that vertical's `tests/`. Commit each vertical's `uv.lock`.
- Colab: `!git clone` (public, no auth), skip the submodule, then `pip install -e ./training` (editable install of the training package).

## Current state (2026-09-12)

- `training/src/training/data/librispeech.py` and the download notebook exist (resumable, md5-verified), plus `training/tests/` (librispeech + notebook regression tests). No feature code yet.
- `fpga/` submodule has the PE / buffer / weight BRAM / LUT modules; array, EPU, FSM, top, and testbenches are stubs.
- `runtime/` and `shared/` are READMEs only.
- Open decisions: hand-RTL vs FINN-GL as the primary bitstream path; weight loading mechanism; GPU compute source; whether sysfs power telemetry is good enough for the report or an external meter is needed.
