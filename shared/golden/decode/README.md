# Decoder golden case

The beam search is written twice, once per vertical, because the verticals share no code: `training/src/training/decode/beam.py` (with the word-LM adapter in `training/src/training/wlm/model.py`) and `runtime/src/runtime/decode/beam.py` (with `runtime/src/runtime/decode/wlm.py`). Training's is the reference. This case is what keeps the two the same.

- `case.json`: the inputs, and what training's search returns for them.
  - `chars`: the vocab the case was written with. Both sides check theirs against it.
  - `am_logp`: AM log-probs `[T, 30]` for a noisy "THE CAT SAT ON THE MAT", with the AM torn between C and H at the start of CAT.
  - `cm_table`: a bigram table `[30, 30]` standing in for the char-LM (`table[prev][next]` is a log-prob). It goes through the same `CharLM` interface as the real CM.
  - `cases`: the search run three ways: no LM, word LM only, both. Each has its `Config`, which LMs it used, and the final N-best list as `[transcript, score]`, best first.
- `words.arpa`: a 7-word trigram, the word LM for the case, read with KenLM as the real one is.

Both sides must return the same transcripts in the same order and the same scores (to a relative 1e-12, which allows for a last-bit difference in `exp`/`log1p` between x86 and the board's libm). The tests are `training/tests/test_decode.py` and `runtime/tests/test_decode.py`.

After any change to the search or the word-LM adapter: make it on both sides, regenerate the case with `python -m training.decode.golden` (in `training/`, with the `wlm` extra), and check that both test suites pass.
