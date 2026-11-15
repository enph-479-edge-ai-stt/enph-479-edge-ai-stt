# hardware/vivado/

Builds the KV260 bitstream from `hardware/rtl/` with two Vivado batch scripts.

- `create_project.tcl`: the project and block design. Zynq PS (KV260 board preset), a simple-mode AXI DMA and `sr_accelerator_axi`, wired as `shared/specs/fabric_interface.md` describes: the PS's HPM0_FPD port writes the DMA's and the accelerator's registers, the DMA's MM2S stream feeds `s_axis`, `m_axis` feeds S2MM, and both DMA memory ports reach DDR through HP0_FPD. One 100 MHz clock (`pl_clk0`) and one reset for everything.
- `build_bitstream.tcl`: synthesis, implementation and the bitstream, then copies `sr_accel.bit` and `sr_accel.hwh` (PYNQ needs both, same base name) plus utilization and timing reports to `<build dir>/out/`.

Register addresses (from the address editor): AXI DMA at `0xA000_0000`, accelerator at `0xA001_0000`. PYNQ reads them from the `.hwh`, so code uses the block names (`axi_dma_0`, `accel`), not the addresses.

## Building

Vivado 2026.1 (needs the free BASIC tier license; 2025.2 and older need none). The build dir must be **outside OneDrive** or any synced folder, since sync locks Vivado's files mid-run. From this folder:

```bash
vivado -mode batch -source create_project.tcl -tclargs ~/vivado-builds/sr_accel 16
vivado -mode batch -source build_bitstream.tcl -tclargs ~/vivado-builds/sr_accel
```

- The second argument of `create_project.tcl` is `PE_COUNT`. It defaults to the real 256. A small one like 16 builds much faster and is enough to test the flow and the board load. The register map is the same; only the row widths in "Loading" shrink.
- The first run on a machine downloads the KV260 board files from AMD's board store.
- On Windows, `vivado` is `C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat`. Each run leaves `vivado.jou` and `vivado.log` in the current folder (gitignored).
- To look at the design, open `<build dir>/sr_accel.xpr` in the Vivado GUI and open the block design.
- A negative worst slack in the build's last line means timing failed at 100 MHz. The bitstream is still written but may misbehave; `out/timing.rpt` has the failing paths.

## Loading on the board

```bash
scp ~/vivado-builds/sr_accel/out/sr_accel.bit ~/vivado-builds/sr_accel/out/sr_accel.hwh kv260:~/
ssh -t kv260 'sudo xmutil listapps'
ssh -t kv260 'sudo /usr/local/share/pynq-venv/bin/python3 -c "from pynq import Overlay; print(Overlay(\"/home/ubuntu/sr_accel.bit\").ip_dict.keys())"'
```

Loading needs sudo. `xmutil listapps` shows whether Kria's own firmware holds the fabric; if one is active, `sudo xmutil unloadapp` frees it first.
