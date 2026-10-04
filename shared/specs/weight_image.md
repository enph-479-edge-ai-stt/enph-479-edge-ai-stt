# Weight image

How a trained model's weights are laid out for the fabric. `training/` writes this layout (`training/src/training/lstm/export.py`), `hardware/` reads it from its weight memory, and `runtime/` streams it in at startup. The acoustic model (AM) and the character model (CM) use the same format: both are a stack of LSTM layers with a linear output layer.

**Status: provisional.** Training writes this layout today. The weight memory, the loader and the controller in `hardware/` are not written yet, so the memory map is open to change until they are. After that, any change here invalidates every exported image and the RTL that reads it.

## Files

Each training run saves two files next to its checkpoints, `am_fabric.*` from the AM notebook and `cm_fabric.*` from the CM notebook:

- `<name>_fabric.mem`: the contents of the weight memory, as text. One 32-bit word per line, 8 hex digits, in the order the board streams them in.
- `<name>_fabric.json`: what the ARM needs for this checkpoint (see "Manifest").

The weights are not part of the bitstream. They need UltraRAM (the AM alone is 8.7 Mb at 6 bits, more than all the block RAM on the KV260), and UltraRAM on Zynq UltraScale+ powers up as zeros with no way to initialize it from a bitstream. So the ARM writes the images in after it loads the bitstream, every time. A new checkpoint needs a new `.mem` file, not a Vivado rebuild.

## Fixed-point formats

| Quantity | Format | Real value |
|---|---|---|
| LSTM weight | 6-bit signed, -31..31 | integer x 2^-7 |
| Output-layer weight | 6-bit signed, -31..31 | integer x 2^-6 |
| Hidden state h | 8-bit signed | integer x 2^-7 |
| AM input feature | 8-bit signed | integer x 2^-5 |
| CM input (one-hot character) | 1 for the current character, 0 elsewhere | integer x 2^0 |
| LSTM accumulator and bias | 24-bit signed | integer x 2^-14 |
| Output-layer accumulator and bias | 24-bit signed | integer x 2^-13 |

A gate's pre-activation is one accumulator: the bias is preloaded, then one product is added per input.

- Every h-phase product, and every product in layers above the first, is h times a weight: 2^-7 x 2^-7 = 2^-14, no shift.
- **Layer 0's x-phase products are shifted left before adding**, because its inputs are not at h's scale. The shift is 7 minus the input's fractional bits (`x_frac` in the manifest):
  - AM: features at 2^-5, shift left 2.
  - CM: a one-hot 1 at 2^0, shift left 7. Only one input is nonzero, so the whole x-phase is one weight row shifted left 7.
- Output layer: its inputs are the top layer's h, products at 2^-6 x 2^-7 = 2^-13, no shift.

`nn.LSTM` has two biases per gate. The image holds their sum, rounded to the accumulator's scale.

The weight steps are powers of two and the same for a layer's input and hidden weights, so the two phases share one accumulator and the rescale in front of the activation is a bit slice.

## Rows

One address holds one row for both PE arrays: PE array 0 in the low half, PE array 1 in the high half. For 256 PEs per array a row is 3072 bits, array 0 in bits `[1535:0]` and array 1 in bits `[3071:1536]`.

- **Weight row:** one input's weight for every PE. Within a half, PE n's weight is at bits `[6n +: 6]`, which is how `pe_array.v` slices its weight inputs.
- **Bias row:** same width, 24-bit fields, so 64 biases per half. A pass's 256 biases take 4 rows; row r holds PEs 64r to 64r + 63, PE n at bits `[24(n - 64r) +: 24]` of its half.

## Memory map

A layer takes two passes, two gates per pass (one per PE array). Each pass is stored as its 4 bias rows followed by one weight row per input, so a pass is read with one incrementing address. Inputs come in this order: the inputs from below (the model's inputs for layer 0, the lower layer's h otherwise), then the layer's own h.

Gates are named as in the RTL: i, f, o, c. The RTL's c is PyTorch's g. Pass 1 is i on PE array 0 and f on PE array 1; pass 2 is o on array 0 and c on array 1.

The output layer comes last, as one more pass: 4 bias rows and one weight row per h. It runs on the first 30 PEs of array 0 (one per vocab symbol) and its rows are zero everywhere else.

Each image has its own addresses, starting at 0.

**AM** (3 layers of 256, 123 features in, 30 out): 3090 rows.

| Address | Rows | Contents |
|---|---|---|
| 0 | 4 + 379 | layer 0, pass 1: x[0..122], then h[0..255] |
| 383 | 4 + 379 | layer 0, pass 2 |
| 766 | 4 + 512 | layer 1, pass 1: lower h[0..255], then own h[0..255] |
| 1282 | 4 + 512 | layer 1, pass 2 |
| 1798 | 4 + 512 | layer 2, pass 1 |
| 2314 | 4 + 512 | layer 2, pass 2 |
| 2830 | 4 + 256 | output layer: top h[0..255] |
| 3090 | | end |

**CM** (2 layers of 256, 30 one-hot characters in, 30 out): 1872 rows.

| Address | Rows | Contents |
|---|---|---|
| 0 | 4 + 286 | layer 0, pass 1: character[0..29], then h[0..255] |
| 290 | 4 + 286 | layer 0, pass 2 |
| 580 | 4 + 512 | layer 1, pass 1: lower h[0..255], then own h[0..255] |
| 1096 | 4 + 512 | layer 1, pass 2 |
| 1612 | 4 + 256 | output layer: top h[0..255] |
| 1872 | | end |

The AM fits one UltraRAM depth (4096 rows): 43 UltraRAMs as one 3072-bit memory, or 44 as two 1536-bit memories on a shared address. The KV260 has 64. **The two images together are 4962 rows, more than one UltraRAM depth,** so the CM cannot simply follow the AM in the same memory. Where the CM's rows go (the block RAM, the UltraRAMs the AM leaves over) is for `hardware/` to decide. It does not change the files.

## Stream order

Rows go in address order. Each row is 96 words, least significant word first, so words 0 to 47 are PE array 0's half and words 48 to 95 are PE array 1's.

| Image | Words | Bytes |
|---|---|---|
| AM | 296,640 | 1,186,560 |
| CM | 179,712 | 718,848 |

- **Simulation:** `$readmemh("am_fabric.mem", words)` into a `logic [31:0] words [0:296639]`, then either drive the loader's stream port with it or assemble rows directly: `row[32*k +: 32] = words[96*addr + k]`.
- **Board:** parse each line as a hex integer into a `uint32` buffer and send it over AXI-DMA. The DMA's "Width of Buffer Length Register" defaults to 14 bits, which truncates a transfer at 16 KB; the AM image needs at least 21.

## Manifest

`<name>_fabric.json` has:

- `n_in`, `n_hidden`, `n_layers`, `n_out`: the model's dimensions.
- `x_frac`: the fractional bits of the model's inputs as the fabric gets them (5 for the AM, 0 for the CM).
- `rows`: the image's size, for checking a load.
- AM only, `cmvn_mean` and `cmvn_std`: 123 values each. The model's inputs are `(features - mean) / std`, and the int8 sent to the fabric is that x 2^`x_frac`, rounded and clamped to -128..127.

## Not specified here yet

The cell state's 16-bit format, the activation tables' input range and size, how h is rounded back to 8 bits, the feature and logit streams, the CM's per-hypothesis state, and the control registers.
