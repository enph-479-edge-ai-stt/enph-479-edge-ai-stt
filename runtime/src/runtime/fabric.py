r"""The fabric driver: load the overlay and the weights, and run jobs (board only, PYNQ).

Registers and streams are ``shared/specs/fabric_interface.md``. Everything here needs
root. As a load check, from ``runtime/src`` on the board:

    sudo /usr/local/share/pynq-venv/bin/python3 -m runtime.fabric sr_accel.bit \
        am_fabric.mem [cm_fabric.mem]

That loads the bitstream, repacks each image (``weights.py``) and streams its four loads
through the AXI DMA, checking LOAD_COUNT after each. The weight memories can't be read
back, so a matching LOAD_COUNT is the check that every row went in.

After the weights, ``Fabric`` runs one job at a time: ``am_frame`` streams an int8
feature frame in and runs the AM on it, ``lm_step`` steps the char LM from one context
slot into another, and both read the 30 int8 scores back over the DMA.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from pynq import Overlay, allocate

from runtime.weights import AM_BASE, Load, cm_base, read_image, repack

CTRL, JOB, BEAM, LM_CHAR, STATUS = 0x00, 0x04, 0x08, 0x0C, 0x10
LOAD_TARGET, LOAD_ADDR, LOAD_COUNT = 0x14, 0x18, 0x1C
SRC_BEAM = 0x28
JOB_AM, JOB_LM, JOB_CLEAR_AM, JOB_CLEAR_BEAM = 0, 1, 2, 3
DONE, ERROR = 0x2, 0x4  # STATUS bits
T_FRAME = 0  # LOAD_TARGET of the frame memory
N_SLOTS = 256  # char-LM context slots (BEAM and SRC_BEAM, 0..255)
N_IN = 123  # AM input features per frame
N_SCORES = 30  # one int8 score per vocab symbol
TIMEOUT_S = 1.0


class Fabric:
    """The accelerator overlay: its registers, the DMA that feeds it and its jobs."""

    def __init__(self, bitfile: str) -> None:
        """Load ``bitfile`` (with its ``.hwh`` next to it) onto the fabric."""
        overlay = Overlay(bitfile)
        self.regs = overlay.accel
        self.dma = overlay.axi_dma_0
        self._frame = allocate(shape=(31,), dtype=np.uint32)  # 124 bytes, 123 used
        self._scores = allocate(shape=(8,), dtype=np.uint32)  # 32 bytes, 30 used

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
        self._run(JOB_CLEAR_AM)

    def clear_slot(self, slot: int) -> None:
        """Zero one char-LM slot's context."""
        self._run(JOB_CLEAR_BEAM, beam=slot)

    def am_frame(self, q: np.ndarray) -> np.ndarray:
        """Run the AM on one int8 feature frame ``[N_IN]``; returns its int8 scores."""
        b = np.zeros(self._frame.nbytes, np.int8)
        b[:N_IN] = q
        self._frame[:] = b.view(np.uint32)
        self._frame.flush()
        self.regs.write(LOAD_TARGET, T_FRAME)
        self.regs.write(LOAD_ADDR, 0)
        self.dma.sendchannel.transfer(self._frame)
        self.dma.sendchannel.wait()
        count = self.regs.read(LOAD_COUNT)
        if count != N_IN:
            raise RuntimeError(f"frame LOAD_COUNT {count}, expected {N_IN}")
        return self._run(JOB_AM, scored=True)

    def lm_step(self, src: int, dst: int, char: int) -> np.ndarray:
        """Step the char LM from slot ``src`` with ``char`` into slot ``dst``.

        Returns the int8 scores. ``src == dst`` steps in place.
        """
        return self._run(JOB_LM, beam=dst, char=char, src=src, scored=True)

    def _run(
        self, job: int, beam: int = 0, char: int = 0, src: int = 0, scored: bool = False
    ) -> np.ndarray | None:
        """Start ``job``, wait for it and, for AM and LM jobs, read its scores back."""
        if scored:
            self.dma.recvchannel.transfer(self._scores)
        self.regs.write(JOB, job)
        self.regs.write(SRC_BEAM, src)
        self.regs.write(BEAM, beam)
        self.regs.write(LM_CHAR, char)
        self.regs.write(CTRL, 1)
        t = time.perf_counter()
        while not self.regs.read(STATUS) & DONE:
            if time.perf_counter() - t > TIMEOUT_S:
                status = self.regs.read(STATUS)
                raise RuntimeError(f"job {job} not done after {TIMEOUT_S} s, STATUS 0x{status:x}")
        if self.regs.read(STATUS) & ERROR:
            raise RuntimeError(f"job {job}: STATUS error bit set")
        if not scored:
            return None
        self.dma.recvchannel.wait()
        self._scores.invalidate()
        return self._scores.view(np.int8)[:N_SCORES].copy()


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
