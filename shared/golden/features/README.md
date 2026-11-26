# Features golden case

Pins what training's feature extraction (`training/src/training/features/fbank.py`) computes, so the runtime's numpy port (`runtime/src/runtime/features.py`), which may not import `training/`, can check it computes the same.

- `6930-75918-0000.flac`: one test-clean utterance, 3.5 s, "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS" (LibriSpeech, CC BY 4.0).
- `case.npz`, written by training from that clip:
  - `feats`: the `[348, 123]` float32 features, before CMVN.
  - `cmvn_mean`, `cmvn_std`: `[123]` float32, the clip's own statistics. They stand in for an AM's: the case pins the arithmetic, not a checkpoint.
  - `x_frac`: the AM's input step, 2^-`x_frac` (5).
  - `int8`: `[348, 123]`, the frames the fabric gets: `(feats - mean) / std * 2^x_frac`, rounded, clamped to -128..127.

The tests are `training/tests/test_features_golden.py` (training's code still writes this case; skips without torch, so not on CI) and `runtime/tests/test_features.py` (the numpy port matches it; on CI). Features must match to 1e-4. The int8 frames must match exactly, except a value within 1e-3 of a rounding tie may land one step either way: training computes in float32 and the port in float64. On this clip that is 1 of 42,804 values.

To see the case and what each test compares on it, run `runtime/prototype/show_features_golden.py` (see `runtime/prototype/README.md`).

After any change to the features, regenerate the case with `python -m training.features.golden` (in `training/`, with torch and torchaudio) and update the numpy port until its test passes.
