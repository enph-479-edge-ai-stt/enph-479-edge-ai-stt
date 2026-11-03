# runtime/prototype/

The term 1 pipeline run on the KV260's CPU before the fabric exists, to measure whether the ARM side keeps up and check that it decodes what Colab decodes:

FLAC -> features -> CMVN -> int8 -> **AM** -> log-softmax -> beam search with the **CM** and KenLM -> text

The two LSTMs (bold) belong to the fabric. Here they are numpy stand-ins, timed apart from the ARM's work. They are a prototype only: the deployed runtime does no RNN math in software.

- `features.py`: the numpy port of `training/src/training/features/fbank.py`, plus CMVN and the int8 rounding the fabric takes. Real runtime code.
- `standin.py`: the AM and the CM as float numpy LSTMs holding the 6-bit weights, as `decode.ipynb` runs them in torch. The CM goes behind the search's `CharLM` interface and counts its steps.
- `export_npz.py`: a checkpoint (`am_6bit.pt`, `cm_6bit.pt`) to the `.npz` the stand-ins load. Runs on a PC with torch, since the board has none.
- `bench.py`: runs and times every stage on the first `--n` of the 500 test-clean utterances `decode.ipynb` evaluates on (same seed), with its tuned weights, and saves the timings and transcripts as JSON.

## Setup on the board

No sudo needed. Everything goes in `~/stt-proto`, the scripts' default `--root`.

```bash
mkdir -p ~/stt-proto/data ~/stt-proto/models && cd ~/stt-proto
git clone https://github.com/enph-479-edge-ai-stt/enph-479-edge-ai-stt.git repo
python3 -m venv venv && . venv/bin/activate
pip install numpy soundfile jiwer pybind11 kenlm==0.3.0   # kenlm builds from source, ~5 min
pip install -e "repo/shared[wlm]"
wget https://files.pythonhosted.org/packages/02/6e/f4e159cef39ea4ef2244b4aa32857d447d60a5d0ee269e1b1f8d3723c03e/kenlm-0.3.0.tar.gz
tar -xzf kenlm-0.3.0.tar.gz && repo/shared/cpp/build.sh kenlm-0.3.0   # the C++ search, ~2 min
```

Data, from OpenSLR's EU mirror (the main server ran at about 22 KB/s from campus on 2026-11-02, the mirror at 4.7 MB/s). The md5s are the ones in `training/src/training/data/librispeech.py`:

```bash
cd ~/stt-proto/data
wget https://openslr.elda.org/resources/12/test-clean.tar.gz https://openslr.elda.org/resources/11/3-gram.pruned.1e-7.arpa.gz
md5sum test-clean.tar.gz 3-gram.pruned.1e-7.arpa.gz   # 32fa31d2..., 9a0f1325...
tar -xzf test-clean.tar.gz && rm test-clean.tar.gz
```

Weights, on a PC with torch, from a training run folder, then copied over:

```bash
python export_npz.py am_6bit.pt am.npz && python export_npz.py cm_6bit.pt cm.npz
scp am.npz cm.npz kv260:stt-proto/models/
```

Then, in `~/stt-proto/repo/runtime/prototype` with the venv active: `python bench.py --n 10`. Long runs go in tmux.

## Results (2026-11-02)

Weights `runs/20261020-0142/am_6bit.pt` and `runs/20261102-0334-cm/cm_6bit.pt`; beam 128, alpha 0.5, beta 0.5, gamma 2.0. The board was idle and logged out of its desktop.

**Same accuracy as Colab.** All 500 utterances (3,599 s of audio) with the C++ search: beam search with both LMs at 24.08% WER and 11.92% CER, equal to `runs/20261102-2019-decode` in every digit it logged. Greedy WER is equal too, and greedy CER is 18.386% against 18.387% (the int8 rounding differences below flip a character or two).

**Correct.** Checked against training's torch code on 3 utterances, with a one-off script kept out of the repo (it imports `training/`, which `runtime/` may not): 4 of 237,513 int8 feature values differ by one step (float64 against float32 rounding at .5), the AM stand-in is within 3.5e-5 of torch in log-prob with the same greedy text, and the CM stand-in within 1.5e-5. On the same 3, the C++ search returned exactly the Python search's N-best and scores, all 128 hypotheses each, also checked with a one-off script.

**Speed**, 10 utterances (60.9 s of audio), one A53 core:

| | Python search | C++ search |
|---|---|---|
| Features + CMVN + int8 + log-softmax | 0.008x real time | 0.007x |
| Beam search, ARM part (with KenLM and the CM's log-softmax) | 0.963x | **0.140x** |
| **ARM total** | **0.970x** | **0.147x** |
| Worst utterance | 1.20x | 0.18x |
| AM stand-in (fabric's work) | 0.278x | 0.274x |
| CM stand-in (fabric's work) | 0.457x | 0.424x |
| Peak RSS | 176 MB | 166 MB |

- Over all 500 utterances with the C++ search the ARM side runs at 0.145x real time: 0.140x at the median utterance, 0.194x at the worst (33.9 s long), none over 0.5x. Peak RSS 205 MB.
- The search was the whole ARM budget in Python. In C++ the ARM side has about 6.8x headroom against real time, on one of the four cores.
- The CM took 1,134 steps per second of audio over the 500 (1,109 over the 10), in 40 batched calls of about 28. At the RTL's cycle counts (about 2,940 per CM step and 4,670 per AM frame, at an assumed 100 MHz) that is about 38 ms of fabric time per second of audio, before the cost of each call's handshake.
