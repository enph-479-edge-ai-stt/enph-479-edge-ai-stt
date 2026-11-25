r"""The fabric driver: load the overlay and stream weight images in (board only, PYNQ).

Registers and streams are ``shared/specs/fabric_interface.md``. Loading needs root. From
``runtime/src`` on the board:

    sudo /usr/local/share/pynq-venv/bin/python3 -m runtime.fabric sr_accel.bit \
        am_fabric.mem [cm_fabric.mem]

That loads the bitstream, repacks each image (``weights.py``) and streams its four loads
through the AXI DMA, checking LOAD_COUNT after each. The weight memories can't be read
back, so a matching LOAD_COUNT is the check that every row went in.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from pynq import Overlay, allocate

from runtime.weights import AM_BASE, Load, cm_base, read_image, repack

LOAD_TARGET = 0x14
LOAD_ADDR = 0x18
LOAD_COUNT = 0x1C
STATUS = 0x10


class Fabric:
    """The accelerator overlay: its registers and the DMA that feeds it."""

    def __init__(self, bitfile: str) -> None:
        """Load ``bitfile`` (with its ``.hwh`` next to it) onto the fabric."""
        overlay = Overlay(bitfile)
        self.regs = overlay.accel
        self.dma = overlay.axi_dma_0

    def load(self, load: Load) -> None:
        """Stream ``load`` into its memory. Raises if LOAD_COUNT doesn't match."""
        self.regs.write(LOAD_TARGET, load.target)
        self.regs.write(LOAD_ADDR, load.addr)
        buf = allocate(shape=load.words.shape, dtype=np.uint32)
        try:
            buf[:] = load.words
            buf.flush()
            self.dma.sendchannel.transfer(buf)
            self.dma.sendchannel.wait()
        finally:
            buf.freebuffer()
        count = self.regs.read(LOAD_COUNT)
        if count != load.rows:
            raise RuntimeError(
                f"target {load.target} from row {load.addr}: LOAD_COUNT {count}, "
                f"expected {load.rows}"
            )


def main() -> None:
    """Load the overlay and the AM's (and optionally the CM's) weight image."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("bitfile")
    p.add_argument("am", help="am_fabric.mem (its .json next to it)")
    p.add_argument("cm", nargs="?", help="cm_fabric.mem (its .json next to it)")
    args = p.parse_args()

    fabric = Fabric(args.bitfile)
    am_image, am = read_image(args.am)
    jobs = [("AM", repack(am_image, am, AM_BASE))]
    if args.cm:
        cm_image, cm = read_image(args.cm)
        jobs.append(("CM", repack(cm_image, cm, cm_base(am))))
    for name, loads in jobs:
        for load in loads:
            t = time.perf_counter()
            fabric.load(load)
            print(
                f"{name} target {load.target}: {load.rows} rows from row {load.addr}, "
                f"{load.words.nbytes} bytes in {time.perf_counter() - t:.3f} s"
            )
    print(f"STATUS 0x{fabric.regs.read(STATUS):08x}")


if __name__ == "__main__":
    main()
