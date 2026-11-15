# runtime/

The live demo program for the KV260's ARM cores (Ubuntu + PYNQ). The program isn't started. Real-time processing is the term 2 goal (January to April); in term 1 the board only has to run the RNN on audio files fed in statically.

`prototype/` runs the term 1 pipeline on the board's CPU before the fabric exists, with numpy stand-ins for the two LSTMs, to measure whether the ARM side keeps up. See its README.

Its first job at startup, after loading the bitstream, is to stream the models' weight images (`am_fabric.mem`, and `cm_fabric.mem` once the char-LM is on the fabric, from the training runs) into the fabric. The format and the load order are in `shared/specs/weight_image.md`.

The decoder it runs on the fabric's output (beam search with the char-LM and the word LM, the C++ build `shared.decode.beam_cpp`) and the vocab are in `shared/` (the `stt-shared` package), not here: training runs the same code on Colab. The runtime takes it as a path dependency on `../shared` once it has a `pyproject.toml`.

## Board setup: PYNQ needs `XILINX_XRT`

Anything that talks to the fabric runs in Kria-PYNQ's venv, `/usr/local/share/pynq-venv` (PYNQ 3.0.1).

**The problem.** PYNQ finds the FPGA only when the `XILINX_XRT` environment variable is set: `pynq/pl_server/__init__.py` imports `EmbeddedDevice`, the Zynq's FPGA, behind `if "XILINX_XRT" in os.environ`. Kria-PYNQ sets it in `/etc/profile.d/pynq_venv.sh`, which only login shells run. So `ssh kv260 '<cmd>'`, `sudo python3` (sudo resets the environment), cron and systemd services all start without it, and PYNQ fails with:

```
UserWarning: No devices found, is the XRT environment sourced?
RuntimeError: No Devices Found
```

**The fix (done 2026-11-14).** A `.pth` file in the venv's `site-packages` sets it every time that interpreter starts. Python runs `import` lines in `.pth` files at startup, so it covers every way of launching the venv's Python, including by path and under `sudo`:

```bash
echo 'import os; os.environ.setdefault("XILINX_XRT", "/usr")' | sudo tee /usr/local/share/pynq-venv/lib/python3.10/site-packages/xilinx_xrt.pth
```

- `setdefault` keeps the profile's value where a login shell already set it (JupyterLab, interactive SSH).
- It only touches the PYNQ venv. Other venvs and the system Python don't see it.
- `bin/activate` would not have worked: nothing runs it when the venv's Python is called by path, and `sudo` would drop it anyway.
- **Rerunning Kria-PYNQ's `install.sh` recreates the venv and deletes the file.** Run the command above again afterwards.

Check it from a non-login shell (prints `/usr`):

```bash
ssh kv260 '/usr/local/share/pynq-venv/bin/python3 -c "import os; print(os.environ.get(\"XILINX_XRT\"))"'
```

Then `prototype/fabric.py` connects to the FPGA without writing to it, and should print `EmbeddedDevice 'KV260'` with the FPGA Manager `operating`.
