# AGENTS.md

Software monorepo for an FPGA speech-recognition system, ENPH 479 capstone (UBC Engineering Physics, 2026W). Team: Kai Asaoka, Sudharshan Kannan, Andrew Du, Anubhav Saini.

Reproduces Lee et al., "FPGA-Based Low-Power Speech Recognition with Recurrent Neural Networks" (arXiv:1610.00552) on a modern board: a deep LSTM acoustic model trained with CTC, a character-level LSTM language model, and a KenLM trigram word LM, fused by N-best beam search. Quantized RNN inference runs on the FPGA fabric of a Kria KV260; everything else runs on the board's ARM cores under Linux.

## Roadmap

The end goal is the live, real-time system. Build for the current term's target first:

- **Term 1 (September to January), current:** the hardware runs the trained RNN on test data fed in statically from audio files. Effectively `model.score()` on audio files, with the fabric doing the RNN math. Offline, no live audio.
- **Term 2 (January to April):** real-time processing (mic capture, streaming, the live demo). Don't build real-time pieces before term 2.

## Three verticals

The repo is split by the three things that get built, each with its own toolchain and its own machine. The verticals share contracts (through `shared/`), not code and not a build environment.

| Vertical | Folder | Runs on | Produces |
|---|---|---|---|
| 1. Training data + ML | `training/` | Google Colab (GPU) / any PC | Quantized weights, packed as weight images for the fabric (`am_fabric.mem` + `am_fabric.json`, and `cm_fabric.*` for the char-LM) |
| 2. Hardware + flashing to FPGA | `hardware/` | Local Linux box with Vivado | Bitstream (`.bit` + `.hwh`) |
| 3. Runtime on the SoM Linux | `runtime/` | KV260 ARM cores, Ubuntu + PYNQ | The live mic-to-browser demo |

`shared/` holds the cross-vertical contracts (feature spec, vocab, fixed-point formats, weight image layout, golden vectors). Nothing in `shared/` belongs to one vertical.

**Hard invariant — no cross-vertical imports.** `training/`, `hardware/`, and `runtime/` must never import from, or otherwise depend on, one another. Anything two verticals both need lives in `shared/`, and in `shared/` only: the verticals depend on `shared/`, never on each other. They run on different machines and Python versions, so a cross-vertical import would not even resolve — this is why, for example, the runtime re-implements the training feature pipeline in numpy and verifies it against `shared/golden/` rather than importing `training`.

```
offline (PC + GPU)
  audio -> features -> LSTM+CTC training -> 6-bit fine-tune -> weight image (am_fabric.mem)
  text  -> char-LM (LSTM) training   -> 6-bit fine-tune -> weight image (cm_fabric.mem)
  text  -> word-LM (KenLM trigram)

offline (Vivado)
  hand-written RTL -> bitstream (.bit + .hwh)

on board (KV260)
  load the bitstream -> stream the weight images into the fabric's weight memory
  USB mic -> numpy features -> [fabric: AM + char-LM] -> beam search + word-LM -> websocket -> browser
```

The model is not compiled into the bitstream. The bitstream is the accelerator; the weights are data it is loaded with at startup.

### 1. `training/` (data + ML, Python)

Python package. Notebooks in `training/notebooks/` are thin launchers only; real code lives in modules so Colab sessions are disposable (clone, run, die).

What exists: `data` (LibriSpeech audio + LM-text download), `features` (123-dim log-mel filterbank + CMVN, computed on the fly), `vocab`, `lstm` (what the two models share: the network, the training loop with its fine-tune on 6-bit weights at the fabric's fixed-point grid, and `export`, which packs a 6-bit model into the weight image), `am` (the acoustic model's own parts: feature input, dataset, CTC loss, greedy CER), `cm` (the character model's own parts, the paper's char-LM: one-hot in, truncated BPTT over an EOS-joined stream of LM-text sentences, bits per character), and two notebooks, `training/notebooks/am_training.ipynb` and `cm_training.ipynb`. Later stages (word-LM, decode, activation and cell quantization, the integer reference model) haven't started; don't scaffold them before they do.

- **One LSTM pipeline.** The AM and the CM are the same kind of network (LSTM stack + linear output layer), so the network class, the training loop, the quantized fine-tune and the export exist once, in `training/src/training/lstm/`. `am/` and `cm/` hold only what differs: how inputs are fed, the loss, and the evaluation. Don't copy pipeline code into a model's folder; if both models need it, it goes in `lstm/`.
- Data: LibriSpeech from OpenSLR 12. Train on `train-clean-100`, tune on `dev-clean`, touch `test-clean` once. The char-LM trains on a random sample of the normalized LibriSpeech LM text (OpenSLR 11, dev/test books excluded) and is scored in bits per character on the dev-clean transcripts.
- Colab: code in GitHub; each run copies the LibriSpeech archives from a cache folder on Google Drive (`MyDrive/enph479-stt/data/`, filled from OpenSLR by the first run) to `/content` scratch, extracts them there, and trains start to finish in one session (no checkpoints, no resume), then saves the final weights (`am.pt` with its CMVN, and `am_6bit.pt` from the 6-bit fine-tune; `cm.pt` and `cm_6bit.pt` for the char-LM), the training logs and the 6-bit model's weight image (`am_fabric.mem` + `am_fabric.json`, or `cm_fabric.*`) to a timestamped run folder on Google Drive.
- Quantization is done by hand on the standard `nn.LSTM`: the fine-tune rounds the weights to the fabric's fixed-point grid every step. No quantization library.
- The weight image's layout and number formats are a contract with `hardware/` and `runtime/`: `shared/specs/weight_image.md`. `lstm/export.py` and that spec change together.
- Stack: PyTorch, torchaudio transforms, soundfile for FLAC I/O, jiwer.
- Keep it lean: build the one path the notebook runs. No fallbacks, no options nothing uses, no code for stages that haven't started.
- The feature pipeline here is the reference the runtime numpy code must bit-match.

### 2. `hardware/` (hardware design + build + flash)

The LSTM accelerator, written by hand in RTL and built into a bitstream with Vivado.

- RTL lives in-repo under `hardware/rtl/`. Hand-written, modelled on the paper's LSTM tile: `pe_unit` (8b x 6b MAC, 24b accumulator), `pe_array`, `pe_buffer` (i/f/o/c gate results), `lstm_epu`, `weight_bram` (packed rows, 6b weights), `sigmoid_lut` / `tanh_lut`, `fsm_controller`, `output_tile`, `sr_accelerator_top`. Several files are still empty stubs. Testbenches in `hardware/testbenches/`.
- The fabric's arithmetic is the reference. The bitstream must match an integer model of that arithmetic bit-for-bit (gate V3 below). That model is not written yet.
- Build machine: Vivado on Linux, ~32 GB RAM, 100+ GB disk. Never on Colab, never on the board. Build outputs (`.bit`, `.xsa`, `.jou`, `.log`, `.Xil/`) are gitignored.
- Weight loading: the weights are written in at startup, not baked into the bitstream. They need UltraRAM (the AM alone is 8.7 Mb at 6 bits, more than all the block RAM), and UltraRAM on this chip powers up as zeros with no bitstream init. After `Overlay()`, the ARM streams the weight images in over AXI-DMA. Swapping checkpoints needs no rebuild. The layout the memory and the controller have to follow is `shared/specs/weight_image.md`; the UltraRAM memory and the loader are not written yet.

### 3. `runtime/` (SoM Linux program)

The live demo on the KV260 ARM cores under Ubuntu + Kria-PYNQ. Not started; real-time processing is the term 2 target (see Roadmap).

Deliberately absent on the board: PyTorch, any RNN math in software. The ARM only sees feature frames going in and per-frame probability vectors coming out.

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
| PL: on-chip SRAM | 26.6 Mb (144 BRAM + 64 URAM blocks) | All weights resident on-chip (~8.8 Mb for the paper's small model at 6 bits) |
| Memory | 4 GB 64-bit DDR4, 16 GB eMMC, QSPI boot flash | Word-LM ARPA lives in DDR4 |
| Carrier I/O | 4x USB 2.0/3.0, 1 GbE, DisplayPort/HDMI, microSD, 12 V jack | USB mic (no analog audio in), ethernet to the browser |

**Flash and boot path.** Boot firmware (FSBL, PMU firmware, U-Boot) lives in QSPI on the SOM itself. The microSD carries Ubuntu 22.04 for Kria; Kria-PYNQ is installed on top (`sudo bash install.sh -b KV260`, JupyterLab on port 9090). The fabric is programmed from Linux, not over JTAG: `pynq.Overlay("design.bit")` calls the kernel FPGA Manager. PYNQ needs the `.bit` and the matching `.hwh` together, always scp both. AMD's `xmutil` is the alternative loader.

**Day-to-day loop:** build on the Linux box -> scp `.bit` + `.hwh` (and the runs' `am_fabric.*` and `cm_fabric.*`) to the board -> ssh or Jupyter -> `Overlay()` -> stream the weight images in over AXI-DMA -> stream frames -> read probabilities -> decode on ARM.

**Reference paper hardware, for comparison** (from the PDF): Xilinx XC7Z045 on a ZC706, 2.18 MB on-chip memory, 512 PEs (two arrays of 256) at 100 MHz, ARM at 800 MHz running the N-best search. 6-bit weights, 8-bit signals, 16-bit LSTM cells. Peephole LSTM. The FPGA ran the small model (3x256 AM, 2x256 char-LM, 6-bit weights, beam 128) at 9.24 W and 4.12x real time, scoring 14.02% WER / 6.02% CER on WSJ eval92. That is this project's target. The headline 8.79% WER / 3.90% CER is the large model (4x512 AM, 2x512 char-LM, 15.1M params) in floating point on a GPU; at ~90 Mb of 6-bit weights it doesn't fit the KV260's 26.6 Mb on-chip either. Power will not be apples-to-apples (28 nm vs 16 nm); frame it as a reproduction on a current platform.

## Cross-vertical contracts (why this is one repo)

The verification gates are bit-exact contracts that cross folder boundaries. Keep them in `shared/` and treat any change there as a breaking change for all three verticals.

- **Features:** 16 kHz mono, 25 ms Hamming window, 10 ms hop, 40 log-mel + energy + delta + double-delta = 123 dims, normalized on training-set statistics, quantized to int8 at the fabric boundary. 100 frames/s. Pin the exact CMVN recipe in `shared/specs/` before anything downstream bakes it in.
- **Vocabulary:** provisional 30 symbols (26 letters, space, apostrophe, end-of-sentence, CTC blank at index 0; LibriSpeech has no other punctuation), in `training/src/training/vocab.py`. Fixed integer mapping once frozen; changing it invalidates every trained artifact.
- **Quantization:** 6-bit weights, 8-bit activations, 16-bit cell state. The weight, bias, input and hidden-state formats are in `shared/specs/weight_image.md`; the cell and activation-table formats are not specified yet.
- **Weight images:** `am_fabric.mem` + `am_fabric.json` and `cm_fabric.mem` + `cm_fabric.json`, written by training, loaded by the runtime, read by the RTL. One format for both models, in `shared/specs/weight_image.md` (provisional until the RTL that reads it exists).
- **Fabric interface:** 123 x int8 in per frame, one int8 per vocab symbol out per frame, over AXI-DMA. Char-LM control (char + context id) over MMIO.
- **Golden vectors:** audio -> features, and the outputs of the integer model of the fabric's arithmetic. The board must match the integer model bit-for-bit.

Gates: V1 greedy CER during float training; V2 CER/WER holds after quantization; V3 board == the integer model, bit-for-bit; V4 end-to-end WER through the board; V5 sustained real-time factor >= 1 and mic-to-screen latency in budget. No power or latency number is reported until V3 passes.

## Git workflow

- **Always create worktrees from `main`.** Never from another feature branch, and never from whatever branch happens to be checked out. Fetch first so `main` is current:
  `git fetch origin && git worktree add ../stt-<name> -b feat/<name> origin/main`
- `main` is the integration branch. Rebase or merge `main` in before opening a PR.
- Before opening a PR, lint and tests must pass in the affected vertical (e.g. `cd training`): `uv run ruff check .` and `uv run pytest` (with `UV_PROJECT_ENVIRONMENT` set — see Environments). CI (`.github/workflows/ci.yml`) runs them on every PR to `main` and on pushes to `main`, for each vertical the change touches. `training` is a required check; it reports as skipped (which counts as passing) when a PR doesn't touch `training/`.
- Branch names: `feat/<thing>`, `fix/<thing>`. One vertical per branch where possible.
- Commit messages follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/#specification): `<type>(<scope>): <description>`, imperative, lowercase, no trailing period. Types: `feat`, `fix`, `docs`, `refactor`, `test`, `build`, `ci`, `chore`. Scope is the vertical or shared area it touches: `training`, `hardware`, `runtime`, `shared`; omit it for repo-wide changes. Anything that changes a contract in `shared/` is a breaking change: add `!` after the scope and a `BREAKING CHANGE:` footer saying which artifacts it invalidates.
- Never commit data, checkpoints, ARPA files, bitstreams, or venvs. `.gitignore` covers them; if you add a new artifact type, add it there.
- Do not commit or push unless asked.

## Environments

- Each vertical is its own **independent uv project** — its own `pyproject.toml`, `.python-version`, `uv.lock`, and venv — because the verticals run on different machines, hardware, and even Python versions. This is deliberately *not* a uv workspace (a workspace shares one lockfile / venv / Python version across members, which is the opposite of what we want). Nothing builds all three at once.
- `training/` is the only Python project today: src layout (package under `training/src/training/`, `import training.data...`), pinned to Python 3.12.0, hatchling backend, with ruff + pytest config and a `dev` dependency group (ruff, pytest) in `training/pyproject.toml`. `runtime/` gets the same treatment when its first module lands; `hardware/` is Verilog, no Python. torch and torchaudio are intentionally not locked (Colab preinstalls them; locally, `uv pip install torch torchaudio` to run the AM tests), so `uv sync` stays fast.
- Venvs must live outside OneDrive (OneDrive evicts package files and corrupts in-folder venvs). This repo is inside OneDrive, so before running any uv command point uv at an external venv by setting `UV_PROJECT_ENVIRONMENT` to an absolute path outside OneDrive (one per vertical) — uv's default in-folder `./.venv` must not be used. Building from the OneDrive-hosted source is fine; only the installed venv needs to sit elsewhere.
- Dev loop (run inside the vertical, e.g. `cd training`, with `UV_PROJECT_ENVIRONMENT` set): `uv sync` creates/updates the venv and installs the `dev` group; `uv run ruff check .` lints (notebooks included), `uv run ruff format` formats, `uv run pytest` runs that vertical's `tests/`. Commit each vertical's `uv.lock`.
- Colab: `!git clone` (public, no auth), then `pip install -e ./training` (editable install of the training package). An editable install registers the package via a `.pth` file that Python only reads at interpreter start, so in the same kernel you must also `sys.path.insert(0, "<repo>/training/src")` (or restart the runtime) before `import training` resolves. The notebook does this.

## Current state (2026-10-03)

- **Decided: hand RTL is the bitstream path.** There is no model compiler in the flow. Training hands the hardware a weight image, and the bit-exact reference for the board is our own integer model.
- `training/`: download, features, vocab, one shared LSTM pipeline (network, training loop, 6-bit weight fine-tune, weight-image export) with the AM and the char-LM on top of it, run from `am_training.ipynb` and `cm_training.ipynb` on Colab (T4). The char-LM's fine-tune and export have not been run on Colab yet. The first full float AM run reached 0.20 greedy dev CER and was still improving (constant learning rate, 20 epochs). The AM tests skip on CI because torch isn't a locked dependency, so CI only covers vocab, download and notebook structure.
- `hardware/`: `pe_unit`, `pe_array`, `pe_buffer`, `lstm_epu`, `weight_bram`, `context_memory` and the two LUTs have content, with testbenches for all but `pe_buffer`. `lstm_tile`, `output_tile`, `fsm_controller` and `sr_accelerator_top` are empty. Not yet in line with the weight image spec: `weight_bram` infers block RAM and is sized for one matrix, `pe_array` defaults its accumulator and bias to 16 bits (the spec needs 24), `lstm_epu` still has a peephole path and placeholder bit slices, and the LUTs are 16-entry tables. There is no loader.
- `shared/`: the weight image spec (provisional). `runtime/` is a README only.
- Not written yet, and needed before gate V3: the integer model of the fabric's arithmetic (8-bit h, 16-bit cell, activation tables).
- Open decisions: sign-off from the hardware side on the weight image's memory map; where the char-LM's rows go on-chip (the two images together are 4962 rows, more than one UltraRAM depth); the cell and activation-table formats; GPU compute source; whether sysfs power telemetry is good enough for the report or an external meter is needed.
