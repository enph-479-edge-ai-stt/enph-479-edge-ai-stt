# runtime/

The live demo program for the KV260's ARM cores (Ubuntu + PYNQ). Real-time processing is the term 2 goal (January to April); in term 1 the board only has to run the RNN on audio files fed in statically.

The `stt-runtime` package (`src/runtime/`) does the term 1 job: audio files to text, with both LSTMs on the fabric and everything else on the ARM.

FLAC -> features -> CMVN -> int8 -> **AM on the fabric** -> log-softmax -> beam search on the ARM with the **char LM on the fabric** and the word LM -> text

- `transcribe.py`: board only, as root. `Transcriber` loads the bitstream, both weight images and the word LM once, then decodes clip after clip with per-stage times. From the command line: `sudo /usr/local/share/pynq-venv/bin/python3 -m runtime.transcribe sr_accel.bit weights/ 3-gram.arpa.gz clip.flac [...] [--json out.json]`, where `weights/` holds `am_fabric.*` and `cm_fabric.*`. The search settings default to decode run 4's (`DECODE_RUN_4`: beam 128, alpha 0.5, beta 0.5, gamma 1.0, tuned with the unpruned trigram).
- `models.py`: the two LSTMs as the decoder sees them. `FabricAcousticModel` runs one AM job per frame, the clip as one batch. `FabricCharModel` is the search's `CharLM`: a state is a `Slot`, one of the fabric's 256 char-LM context slots, and a step is one LM job that reads the parent's slot and writes a free one, each search step's children as one batch. A slot goes back to the `SlotPool` when the search drops the last prefix holding it. When the search steps the char LM, the live states are the last beam's (up to 128) plus the new children (up to 128), so 256 always suffice, which needs `beam_cpp` to free the root prefix like any other (it does). No pynq import, so the tests run it on a stand-in fabric.
- `fabric.py`: board only. Loads the overlay and the weight images (each of the four loads streamed through the AXI DMA, LOAD_COUNT checked after each since the weight memories can't be read back), then runs jobs in batches through the C++ job loop (`cpp/jobs.cpp`, built as `runtime._jobs`): `am(q)` streams each int8 frame in and runs the AM on it, `lm(src, dst, chars)` steps the char LM from one slot into another for each entry (SRC_BEAM and BEAM, `shared/specs/fabric_interface.md`), and both read every job's 30 int8 scores back. Python hands over a batch (a clip's frames, or one search step's children), so a job costs its fabric time plus a few register writes instead of a Python round trip. As a load check, from `runtime/src`: `sudo /usr/local/share/pynq-venv/bin/python3 -m runtime.fabric sr_accel.bit am_fabric.mem [cm_fabric.mem]`. Needs a full-size (`PE_COUNT` 256) bitstream with SRC_BEAM (main since PR #49).
- `weights.py`: reads a `.mem` image and its `.json` manifest (`shared/specs/weight_image.md`) and repacks it into the four loads the RTL takes (`shared/specs/fabric_interface.md`, "Loading"): LSTM weight rows as they are, each pass's 4 bias rows joined into one row per pass, and the output layer's weights and biases cut down to its 30 PEs. The CM's rows follow the AM's. Pure numpy, runs anywhere.
- `features.py`: the numpy port of training's feature extraction, plus CMVN and the int8 rounding the fabric takes. Held to training's by `shared/golden/features/` (`tests/test_features.py`): features within 1e-4, int8 frames equal except at rounding ties.
- `tests/` (`uv run pytest` in this folder, no board): the repack against an image built straight from the spec, the features against training's golden case, and `models.py` on a stand-in fabric through `shared.decode.beam` (the Python search, same interface as `beam_cpp`): same transcript as the stand-in char LM called directly, and every slot free afterwards.

The decoder (beam search with the char LM and the word LM, built in C++ as `shared.decode.beam_cpp`) and the vocab are in `shared/` (the `stt-shared` package), a path dependency on `../shared`: training runs the same code on Colab.

`prototype/` holds the checks that got here: the CPU stand-ins and timing (`bench.py`), the integer model that is the bit-exact reference for the board (`fixed.py`), and the board checks against it (`predict_one.py`, `fabric_am.py`, `fork_test.py`, and `fabric_decode.py`, the 500-clip check of `transcribe.py`). The package does no RNN math in software; `fixed.py` stays out of it. See its README.

## Board setup: installing the runtime into PYNQ's venv

The fabric needs root and PYNQ, so the runtime runs in Kria-PYNQ's venv as root. Point that venv at the two packages' `src/` folders in a checkout of this repo with a `.pth` file, add `soundfile` (FLAC input, which the venv doesn't have), and build the C++ search with the venv's Python (pybind11 is already in it). With the checkout in `~/stt-repo` and the kenlm 0.3.0 source unpacked as in `prototype/README.md`:

```bash
printf '%s\n' ~/stt-repo/shared/src ~/stt-repo/runtime/src \
    | sudo tee /usr/local/share/pynq-venv/lib/python3.10/site-packages/stt.pth > /dev/null
sudo /usr/local/share/pynq-venv/bin/pip install soundfile
PYTHON=/usr/local/share/pynq-venv/bin/python3 bash ~/stt-repo/shared/cpp/build.sh ~/stt-proto/kenlm-src/kenlm-0.3.0
PYTHON=/usr/local/share/pynq-venv/bin/python3 bash ~/stt-repo/runtime/cpp/build.sh   # the job loop, seconds
sudo /usr/local/share/pynq-venv/bin/python3 -c "import runtime.transcribe; print('ok')"
```

- **Why not `pip install -e`:** the venv's pip (22.0.2) runs hatchling in an isolated build that still picks up the venv's old `packaging`, and fails with `No module named 'packaging.licenses'` (2026-11-26). Pinning hatchling below 1.27 didn't help, and upgrading `packaging` or pip would change the venv PYNQ itself runs on. The `.pth` file is what an editable install writes anyway, without the build. numpy stays the venv's (1.26.4).
- The checkout has to have LF line endings: `build.sh` fails with `set: pipefail: invalid option name` if it was copied over from a Windows working tree.
- An update is a `git pull`, plus a rebuild of `beam_cpp` when `shared/cpp/` changes and of the job loop when `runtime/cpp/` changes. **Rerunning Kria-PYNQ's `install.sh` recreates the venv**, which drops `stt.pth` and `soundfile` along with the `.pth` fix below, so redo them.

## Board setup: the PYNQ venv's environment

Anything that talks to the fabric runs in Kria-PYNQ's venv, `/usr/local/share/pynq-venv` (PYNQ 3.0.1). PYNQ depends on two things that Kria-PYNQ only sets up in `/etc/profile.d/pynq_venv.sh`, which only login shells run. So `ssh kv260 '<cmd>'`, `sudo python3` (sudo resets the environment and `PATH`), cron and systemd services all start without them. JupyterLab and interactive SSH logins were fine, which is why this hid until 2026-11-14.

**1. `XILINX_XRT`.** `pynq/pl_server/__init__.py` imports `EmbeddedDevice`, the Zynq's FPGA, only `if "XILINX_XRT" in os.environ`. Without it:

```
UserWarning: No devices found, is the XRT environment sourced?
RuntimeError: No Devices Found
```

**2. The venv's `bin` on `PATH`.** To load a `.bit`, PYNQ runs plain `xclbinutil` to build a temporary `.xclbin`. There are two on the board: Kria-PYNQ's installer puts a working one in the venv (`/usr/local/share/pynq-venv/bin/xclbinutil`), and Ubuntu's `xrt` package has `/usr/bin/xclbinutil`, which segfaults at startup (even `--help`). Without the venv on `PATH`, PYNQ gets Ubuntu's. Its wrapper script exits 0 anyway, so the error shows up later as:

```
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/tmpXXXXXXXX/t.xclbin'
```

**The fix (done 2026-11-14).** One `.pth` file in the venv's `site-packages` sets both every time that interpreter starts, the way activating the venv would. Python runs `import` lines in `.pth` files at startup, so it covers every way of launching the venv's Python, including by path and under `sudo`. On the board:

```bash
sudo tee /usr/local/share/pynq-venv/lib/python3.10/site-packages/pynq_env.pth > /dev/null <<'EOF'
import os, sys; os.environ.setdefault("XILINX_XRT", "/usr"); _bin = os.path.join(sys.prefix, "bin"); os.environ["PATH"] = os.environ["PATH"] if _bin in os.environ.get("PATH", "").split(os.pathsep) else _bin + os.pathsep + os.environ.get("PATH", "")
EOF
```

- `setdefault` and the membership check leave a login shell's values alone (JupyterLab, interactive SSH).
- It only touches the PYNQ venv. Other venvs and the system Python don't see it.
- `bin/activate` would not have worked: nothing runs it when the venv's Python is called by path, and `sudo` would drop it anyway.
- **Rerunning Kria-PYNQ's `install.sh` recreates the venv and deletes the file.** Run the command above again afterwards.

Check it from a non-login shell (prints `/usr` and the venv's `xclbinutil`):

```bash
ssh kv260 '/usr/local/share/pynq-venv/bin/python3 -c "import os, shutil; print(os.environ.get(\"XILINX_XRT\"), shutil.which(\"xclbinutil\"))"'
```

Then `prototype/fabric.py` connects to the FPGA without writing to it, and should print `EmbeddedDevice 'KV260'` with the FPGA Manager `operating`. With the fix in, `sudo /usr/local/share/pynq-venv/bin/python3` loads overlays from plain SSH (first done 2026-11-14 with the `hardware/vivado/` bitstream).
