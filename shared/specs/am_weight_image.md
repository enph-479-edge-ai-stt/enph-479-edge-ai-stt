# AM weight image

How the trained acoustic model's weights are laid out for the fabric. `training/` writes this layout (`training/src/training/am/export.py`), `hardware/` reads it from its weight memory, and `runtime/` streams it in at startup.

**Status: provisional.** Training writes this layout today. The weight memory, the loader and the controller in `hardware/` are not written yet, so the memory map is open to change until they are. After that, any change here invalidates every exported image and the RTL that reads it.

## Files

Each training run saves two files next to its checkpoints:

- `am_fabric.mem`: the contents of the weight memory, as text. One 32-bit word per line, 8 hex digits, in the order the board streams them in.
- `am_fabric.json`: what the ARM needs for this checkpoint (see "Manifest").

The weights are not part of the bitstream. They need UltraRAM (8.7 Mb at 6 bits, more than all the block RAM on the KV260), and UltraRAM on Zynq UltraScale+ powers up as zeros with no way to initialize it from a bitstream. So the ARM writes the image in after it loads the bitstream, every time. A new checkpoint needs a new `am_fabric.mem`, not a Vivado rebuild.

## Fixed-point formats

| Quantity | Format | Real value |
|---|---|---|
| LSTM weight | 6-bit signed, -31..31 | integer x 2^-7 |
| Output-layer weight | 6-bit signed, -31..31 | integer x 2^-6 |
| Hidden state h | 8-bit signed | integer x 2^-7 |
| Input feature | 8-bit signed | integer x 2^-5 |
| LSTM accumulator and bias | 24-bit signed | integer x 2^-14 |
| Output-layer accumulator and bias | 24-bit signed | integer x 2^-13 |

A gate's pre-activation is one accumulator: the bias is preloaded, then one product is added per input.

- Layers 1 and 2: every input is an h (the layer below's, then the layer's own), so every product is already at 2^-7 x 2^-7 = 2^-14.
- Layer 0: its first 123 inputs are features at 2^-5, so those products are at 2^-12. **Shift them left by 2 bits before adding.** Its h-phase products need no shift.
- Output layer: its inputs are the top layer's h, products at 2^-6 x 2^-7 = 2^-13, no shift.

`nn.LSTM` has two biases per gate. The image holds their sum, rounded to the accumulator's scale.

The weight steps are powers of two and the same for a layer's input and hidden weights, so the two phases share one accumulator and the rescale in front of the activation is a bit slice.

## Rows

One address holds one row for both PE arrays: PE array 0 in the low half, PE array 1 in the high half. For 256 PEs per array a row is 3072 bits, array 0 in bits `[1535:0]` and array 1 in bits `[3071:1536]`.

- **Weight row:** one input's weight for every PE. Within a half, PE n's weight is at bits `[6n +: 6]`, which is how `pe_array.v` slices its weight inputs.
- **Bias row:** same width, 24-bit fields, so 64 biases per half. A pass's 256 biases take 4 rows; row r holds PEs 64r to 64r + 63, PE n at bits `[24(n - 64r) +: 24]` of its half.

## Memory map

A layer takes two passes, two gates per pass (one per PE array). Each pass is stored as its 4 bias rows followed by one weight row per input, so a pass is read with one incrementing address. Inputs come in this order: the inputs from below (features for layer 0, the lower layer's h otherwise), then the layer's own h.

Gates are named as in the RTL: i, f, o, c. The RTL's c is PyTorch's g.

| Address | Rows | Contents | PE array 0 | PE array 1 |
|---|---|---|---|---|
| 0 | 4 + 379 | layer 0, pass 1: x[0..122], then h[0..255] | i | f |
| 383 | 4 + 379 | layer 0, pass 2 | o | c |
| 766 | 4 + 512 | layer 1, pass 1: lower h[0..255], then own h[0..255] | i | f |
| 1282 | 4 + 512 | layer 1, pass 2 | o | c |
| 1798 | 4 + 512 | layer 2, pass 1 | i | f |
| 2314 | 4 + 512 | layer 2, pass 2 | o | c |
| 2830 | 4 + 256 | output layer: top h[0..255] | PEs 0..29 | zeros |
| 3090 | | end | | |

The output layer runs on the first 30 PEs of array 0 (one per vocab symbol). Its rows are zero everywhere else.

3090 rows fit one UltraRAM depth (4096). As one 3072-bit memory that is 43 UltraRAMs; as two 1536-bit memories on a shared address, 44. The KV260 has 64.

## Stream order

Rows go in address order. Each row is 96 words, least significant word first, so words 0 to 47 are PE array 0's half and words 48 to 95 are PE array 1's. The whole image is 296,640 words, 1,186,560 bytes.

- **Simulation:** `$readmemh("am_fabric.mem", words)` into a `logic [31:0] words [0:296639]`, then either drive the loader's stream port with it or assemble rows directly: `row[32*k +: 32] = words[96*addr + k]`.
- **Board:** parse each line as a hex integer into a `uint32` buffer and send it over AXI-DMA. The DMA's "Width of Buffer Length Register" defaults to 14 bits, which truncates a transfer at 16 KB; this image needs at least 21.

## Manifest

`am_fabric.json` has:

- `cmvn_mean`, `cmvn_std`: 123 values each. The model's inputs are `(features - mean) / std`.
- `x_frac`: the int8 feature sent to the fabric is the normalized feature x 2^`x_frac`, rounded and clamped to -128..127.
- `n_feats`, `n_hidden`, `n_layers`, `n_out`: the model's dimensions.
- `weight_bits`, `lstm_frac`, `proj_frac`, `h_frac`, `bias_bits`: the formats above, as the image was built.
- `rows`, `words_per_row`: the image's size, for checking a load.

## Not specified here yet

The cell state's 16-bit format, the activation tables' input range and size, how h is rounded back to 8 bits, the feature and logit streams, and the control registers.
