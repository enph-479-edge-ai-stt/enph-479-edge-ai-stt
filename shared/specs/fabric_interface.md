# Fabric interface

How the ARM drives the accelerator: the control registers and the two AXI streams of `hardware/rtl/sr_accelerator_axi.v`. `hardware/` implements it, `runtime/` codes against it.

**Status: provisional.** It follows the RTL as it is today, which keeps the LSTM weights, the LSTM biases, the output-layer weights and the output-layer biases in four separate memories rather than in the one image of `weight_image.md`. Until the RTL takes that layout, the ARM repacks each `.mem` file into the four memories' rows before streaming it (see "Loading"). When it does, the load targets collapse to one and this file changes with it.

## Connections

| Port | AXI | To |
|---|---|---|
| `s_axil` | AXI-Lite slave, 32-bit, 64-byte register space | the PS's general-purpose master port |
| `s_axis` | AXI-Stream slave, 32-bit | AXI DMA MM2S (memory to fabric) |
| `m_axis` | AXI-Stream master, 32-bit, with `tlast` | AXI DMA S2MM (fabric to memory) |

`clk` is the fabric clock (100 MHz target) and `rst_n` its active-low reset. The DMA's "Width of Buffer Length Register" must be at least 21 bits so a whole weight load fits in one transfer.

## Registers

| Offset | Name | Access | Bits |
|---|---|---|---|
| 0x00 | CTRL | write | bit 0: 1 starts a job |
| 0x04 | JOB | read/write | 0 AM, 1 LM, 2 clear AM context, 3 clear one beam's LM context |
| 0x08 | BEAM | read/write | the LM job's (or beam clear's) beam, 0..127 |
| 0x0C | LM_CHAR | read/write | the LM job's input character, a vocab index |
| 0x10 | STATUS | read | bit 0 busy, bit 1 done, bit 2 error, bit 3 scores waiting on `m_axis` |
| 0x14 | LOAD_TARGET | read/write | which memory `s_axis` fills (see "Loading") |
| 0x18 | LOAD_ADDR | read/write | the first row (or frame byte) that `s_axis` writes |
| 0x1C | LOAD_COUNT | read | rows (or frame bytes) written since the last LOAD_TARGET or LOAD_ADDR write |
| 0x20 | CTX_ADDR | read/write | a context-memory entry to read back |
| 0x24 | CTX_DATA | read | that entry, `{c[15:0], h[7:0]}` in bits 23:0 |

JOB, BEAM and LM_CHAR are captured when CTRL starts the job, so the next job can be set up while one runs. STATUS.done stays set until the next start.

A start is ignored while the previous job runs (busy) and while its scores still wait on `m_axis`; read STATUS to see whether it took. A start that arrives while the last loaded row is still being written (a few cycles after its last word) waits for the write and then runs.

## Loading

Writing LOAD_TARGET or LOAD_ADDR restarts the loader at row LOAD_ADDR. A row arrives as 32-bit words, least significant word first; once its last word is in, it is written and the next row follows. LOAD_COUNT counts a row once it is written, a cycle after its last word (a frame word's 4 bytes take 4 cycles). The stream stalls while a job runs.

| LOAD_TARGET | Memory | Row | Words per row |
|---|---|---|---|
| 0 | input frame | 4 int8 features per word, feature 0 in the low byte; bytes past feature 122 are dropped | 1 |
| 1 | LSTM weights | 6-bit weights: PE array 0's 256 in the low half, array 1's in the high half, PE n at bits `[6n +: 6]` of its half (`weight_image.md`'s weight row) | 96 |
| 2 | LSTM biases | 24-bit biases: array 0's 256 in the low half, array 1's in the high half, PE n at bits `[24n +: 24]` of its half | 384 |
| 3 | output-layer weights | 30 6-bit weights, symbol v at bits `[6v +: 6]` | 6 |
| 4 | output-layer biases | 30 24-bit biases, symbol v at bits `[24v +: 24]` | 23 |

Row addresses, as the controller reads them:

- **LSTM weights:** each layer's two passes back to back, each pass its weight rows for the inputs from below then its own h (`weight_image.md`'s pass, without the 4 bias rows). The AM's layers first from row 0, then the CM's from row 2806.
- **LSTM biases:** row `2 * layer + pass`, the AM's 3 layers as layers 0..2 and the CM's 2 as layers 3..4.
- **Output-layer weights:** one row per input h, the AM's rows 0..255, the CM's 256..511.
- **Output-layer biases:** row 0 the AM's, row 1 the CM's.

## Scores

An AM or LM job sends its 30 int8 scores on `m_axis` once it is done: 8 words, 4 scores per word with symbol 0 in the low byte of the first word, `tlast` on the eighth. A score is the logit at 2^-2, saturated (`weight_image.md`, "Activations and cell"), so the logit is the score / 4. Clear jobs send nothing. Start the S2MM transfer before or after the job; the scores wait until they are taken.
