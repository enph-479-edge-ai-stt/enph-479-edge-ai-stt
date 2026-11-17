# runtime/

The live demo program for the KV260's ARM cores (Ubuntu + PYNQ). Real-time processing is the term 2 goal (January to April); in term 1 the board only has to run the RNN on audio files fed in statically.

`prototype/` runs the term 1 pipeline on the board's CPU before the fabric exists, with numpy stand-ins for the two LSTMs, to measure whether the ARM side keeps up. See its README.

The runtime's first job at startup, after loading the bitstream, is to stream the models' weight images (`am_fabric.mem` and `cm_fabric.mem` from the training runs) into the fabric. That part is written, in `src/runtime/` (the `stt-runtime` package):

- `weights.py`: reads a `.mem` image and its `.json` manifest (`shared/specs/weight_image.md`) and repacks it into the four loads the RTL takes (`shared/specs/fabric_interface.md`, "Loading"): LSTM weight rows as they are, each pass's 4 bias rows joined into one row per pass, and the output layer's weights and biases cut down to its 30 PEs. The CM's rows follow the AM's. Pure numpy, runs anywhere.
- `fabric.py`: board only. Loads the overlay and streams each load through the AXI DMA, checking LOAD_COUNT after each (the weight memories can't be read back). As root, from `runtime/src`: `sudo /usr/local/share/pynq-venv/bin/python3 -m runtime.fabric sr_accel.bit am_fabric.mem [cm_fabric.mem]`. Needs a full-size (`PE_COUNT` 256) bitstream; a smaller test build has narrower rows than the images. First run 2026-11-14, `hardware/vivado/` build at PE_COUNT 256 with AM run `20261103-2314` and CM run `20261102-0334-cm`: all 8 loads' LOAD_COUNTs matched, about 1.7 MB in under 20 ms (the AM's 1.08 MB of LSTM weights in 7 ms).
- `features.py`: the numpy port of training's feature extraction, plus CMVN and the int8 rounding the fabric takes. Held to training's by `shared/golden/features/` (`tests/test_features.py`): features within 1e-4, int8 frames equal except at rounding ties.
- `tests/`: the repack against an image built straight from the spec (`uv run pytest` in this folder). It was also checked once against `training/`'s `export.py` at the real AM and CM sizes, with a script kept out of the repo since `runtime/` may not import `training/`.

The decoder it runs on the fabric's output (beam search with the char-LM and the word LM, the C++ build `shared.decode.beam_cpp`) and the vocab are in `shared/` (the `stt-shared` package), not here: training runs the same code on Colab. The runtime will take it as a path dependency on `../shared` when it starts decoding.

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
