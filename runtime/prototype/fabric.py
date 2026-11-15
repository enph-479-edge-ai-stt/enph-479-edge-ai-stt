"""Connect to the KV260's FPGA through PYNQ, without touching the fabric.

The first piece of the fabric driver: it finds the FPGA the way ``pynq.Overlay`` will and
reports its state. Nothing is written. No bitstream is loaded, no register is read or
written and no memory is allocated, so it runs as the ``ubuntu`` user without sudo.

PYNQ finds the FPGA only when XILINX_XRT is set. The board's PYNQ venv sets it at startup
(``site-packages/xilinx_xrt.pth``), so this runs from any shell:

    /usr/local/share/pynq-venv/bin/python3 fabric.py
"""

from __future__ import annotations

import os
import sys

import pynq
from pynq.pl_server import global_state

FPGA_MANAGER = "/sys/class/fpga_manager/fpga0"


def connect() -> pynq.Device:
    """The board's FPGA as PYNQ sees it. Raises if PYNQ finds none."""
    return pynq.Device.active_device


def main() -> int:
    device = connect()
    with open(f"{FPGA_MANAGER}/state") as f:
        state = f.read().strip()
    loaded = global_state.load_global_state() if global_state.global_state_file_exists() else None
    can_load = os.access(f"{FPGA_MANAGER}/firmware", os.W_OK) and os.access("/dev/mem", os.W_OK)

    print(f"pynq           {pynq.__version__}")
    print(f"device         {type(device).__name__} {device.name!r}")
    print(f"fpga manager   {state}")
    print(f"pynq overlay   {f'{loaded.bitfile_name} ({loaded.timestamp})' if loaded else 'none loaded'}")
    print(f"can load here  {'yes' if can_load else 'no, overlay loads need sudo'}")
    return 0 if state == "operating" else 1


if __name__ == "__main__":
    sys.exit(main())
