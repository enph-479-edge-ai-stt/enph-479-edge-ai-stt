# runtime/

The live demo program for the KV260's ARM cores (Ubuntu + PYNQ). The program isn't started. Real-time processing is the term 2 goal (January to April); in term 1 the board only has to run the RNN on audio files fed in statically.

`prototype/` runs the term 1 pipeline on the board's CPU before the fabric exists, with numpy stand-ins for the two LSTMs, to measure whether the ARM side keeps up. See its README.

Its first job at startup, after loading the bitstream, is to stream the models' weight images (`am_fabric.mem`, and `cm_fabric.mem` once the char-LM is on the fabric, from the training runs) into the fabric. The format and the load order are in `shared/specs/weight_image.md`.

The decoder it runs on the fabric's output (beam search with the char-LM and the word LM, the C++ build `shared.decode.beam_cpp`) and the vocab are in `shared/` (the `stt-shared` package), not here: training runs the same code on Colab. The runtime takes it as a path dependency on `../shared` once it has a `pyproject.toml`.

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
