r"""The fabric driver: load the overlay and the weights, and run jobs (board only, PYNQ).

Registers and streams are ``shared/specs/fabric_interface.md``. Everything here needs
root. As a load check, from ``runtime/src`` on the board:

    sudo /usr/local/share/pynq-venv/bin/python3 -m runtime.fabric sr_accel.bit \
        am_fabric.mem [cm_fabric.mem]

That loads the bitstream, repacks each image (``weights.py``) and streams its four loads
through the AXI DMA, checking LOAD_COUNT after each. The weight memories can't be read
back, so a matching LOAD_COUNT is the check that every row went in.

After the weights, ``Fabric`` runs jobs in batches through the C++ job loop
(``runtime._jobs``, built by ``runtime/cpp/build.sh``): ``am`` streams each int8 feature
frame in and runs the AM on it, ``lm`` steps the char LM from one context slot into
another for each (src, dst, char), and both read every job's 30 int8 scores back over the
DMA. Python only hands over the batch, so a job costs its fabric time plus a few register
writes, not a Python round trip.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from pynq import Overlay, allocate

from runtime import _jobs
from runtime.weights import AM_BASE, Load, cm_base, read_image, repack

STATUS = 0x10
LOAD_TARGET, LOAD_ADDR, LOAD_COUNT = 0x14, 0x18, 0x1C
JOB_CLEAR_AM, JOB_CLEAR_BEAM = 2, 3  # the jobs' other registers are in cpp/jobs.cpp
N_SLOTS = 256  # char-LM context slots (BEAM and SRC_BEAM, 0..255)


class Fabric:
    """The accelerator overlay: its registers, the DMA that feeds it and its jobs."""

    def __init__(self, bitfile: str) -> None:
        """Load ``bitfile`` (with its ``.hwh`` next to it) onto the fabric."""
        overlay = Overlay(bitfile)
        self.regs = overlay.accel
        self.dma = overlay.axi_dma_0
        # Non-cacheable, so the C++ loop needs no cache maintenance around the DMA.
        self._frame = allocate(shape=(31,), dtype=np.uint32, cacheable=False)  # 123 of 124 B
        self._scores = allocate(shape=(8,), dtype=np.uint32, cacheable=False)  # 30 of 32 B
        self._jobs = _jobs.Jobs(
            self.regs.mmio.array,
            self.dma.mmio.array,
            self._frame,
            self._frame.physical_address,
            self._scores,
            self._scores.physical_address,
        )

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

    def load_images(self, am_mem: str | Path, cm_mem: str | Path) -> tuple[dict, dict]:
        """Stream both models' weight images in; returns their manifests."""
        am_image, am = read_image(am_mem)
        cm_image, cm = read_image(cm_mem)
        for load in repack(am_image, am, AM_BASE) + repack(cm_image, cm, cm_base(am)):
            self.load(load)
        return am, cm

    def clear_am(self) -> None:
        """Zero the AM's context, as at the start of an utterance."""
        self._jobs.clear(JOB_CLEAR_AM)

    def clear_slot(self, slot: int) -> None:
        """Zero one char-LM slot's context."""
        self._jobs.clear(JOB_CLEAR_BEAM, slot)

    def am(self, q: np.ndarray) -> np.ndarray:
        """Run the AM on each int8 feature frame ``[T, 123]`` in turn: int8 scores ``[T, 30]``."""
        return self._jobs.am(q)

    def lm(self, src, dst, chars) -> np.ndarray:
        """Step the char LM: slot ``dst[k]`` = step(slot ``src[k]``, ``chars[k]``), in order.

        Returns the int8 scores ``[N, 30]``. ``src[k] == dst[k]`` steps in place.
        """
        return self._jobs.lm(src, dst, chars)


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
