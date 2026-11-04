# Decoder golden case

Pins the output of the beam search in `shared/src/shared/decode/beam.py` (with the word-LM adapter in `decode/wlm.py`), and so of its C++ port in `shared/cpp/`, which training and the board run. This case is what shows the two agree, and that the board decodes the same as the PC: same N-best, same scores.

- `case.json`: the inputs, and what the search returns for them.
  - `chars`: the vocab the case was written with. The tests check `shared.vocab` against it.
  - `am_logp`: AM log-probs `[T, 30]` for a noisy "THE CAT SAT ON THE MAT", with the AM torn between C and H at the start of CAT.
  - `cm_table`: a bigram table `[30, 30]` standing in for the char-LM (`table[prev][next]` is a log-prob). It goes through the same `CharLM` interface as the real CM.
  - `cases`: the search run three ways: no LM, word LM only, both. Each has its `Config`, which LMs it used, and the final N-best list as `[transcript, score]`, best first.
- `words.arpa`: a 7-word trigram, the word LM for the case, read with KenLM as the real one is.

The scores must match to a relative 1e-12, which allows for a last-bit difference in `exp`/`log1p` between x86 and the board's libm. The tests are `shared/tests/test_decode.py` (Python) and `shared/tests/test_beam_cpp.py` (C++).

After any change to the search or the word-LM adapter, regenerate the case with `python -m shared.decode.golden` (in `shared/`, with the `wlm` extra) and check the tests pass.
