"""Show the features golden case: what its tests take in, what they compare, and a figure.

Prints the clip and ``case.npz``'s arrays, then each test of ``tests/test_features.py``
worked through on real values (one frame side by side, the largest differences per
feature group, every int8 value that differs and how close it sat to a tie). Saves a
figure: the waveform, training's log-mel features, the int8 frames the fabric gets, and
|runtime - training| per value. Needs matplotlib, which isn't a dependency. From
``runtime/``:

    uv run --with matplotlib python prototype/show_features_golden.py [out.png]

The figure goes to ``prototype/show_features_golden.png`` (gitignored) unless given a path.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from runtime.features import X_FRAC, Fbank

GOLDEN = Path(__file__).resolve().parents[2] / "shared" / "golden" / "features"
CLIP = GOLDEN / "6930-75918-0000.flac"
GROUPS = {
    "log-mel 0-39": slice(0, 40),
    "energy 40": slice(40, 41),
    "delta 41-81": slice(41, 82),
    "delta-delta 82-122": slice(82, 123),
}
np.set_printoptions(precision=4, suppress=True, linewidth=110)


def rule(title):
    """Print a section heading."""
    print(f"\n{'=' * 8} {title} {'=' * (70 - len(title))}")


# ---------------------------------------------------------------- inputs
wav, sr = sf.read(str(CLIP), dtype="float32")
case = np.load(GOLDEN / "case.npz")
rule("INPUT 1: the audio clip")
print(f"{CLIP.name}: {sr} Hz, {len(wav)} samples = {len(wav) / sr:.3f} s")
print('transcript: "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS"')
print(f"peak |sample| {np.abs(wav).max():.4f}; samples 8000..8007: {wav[8000:8008]}")

rule("INPUT 2: case.npz, written by training (torch)")
for k in case.files:
    v = case[k]
    print(f"  {k:10s} {str(v.dtype):8s} shape {v.shape}" + (f" = {int(v)}" if v.ndim == 0 else ""))

# ---------------------------------------------------------------- what the tests compute
fb = Fbank(case["cmvn_mean"], case["cmvn_std"])
feats = fb.extract(wav)
q = fb.int8(feats)
want_f, want_q = case["feats"], case["int8"]
x = (feats - fb.mean) / fb.std * 2.0**X_FRAC  # the unrounded int8 value

rule("TEST test_x_frac_is_the_cases")
print(
    f"case x_frac = {int(case['x_frac'])}, runtime X_FRAC = {X_FRAC}  ->  "
    f"{'PASS' if int(case['x_frac']) == X_FRAC else 'FAIL'}"
)

rule("TEST test_features_match_training  (tolerance 1e-4)")
t = 150
print(f"frame {t}, first 6 log-mel dims:")
print(f"  training (torch, float32): {want_f[t, :6]}")
print(f"  runtime  (numpy, float64): {feats[t, :6]}")
d = np.abs(feats - want_f)
for name, s in GROUPS.items():
    print(f"  {name:20s} max |diff| {d[:, s].max():.2e}")
print(f"  overall max |diff| {d.max():.2e}  ->  {'PASS' if d.max() <= 1e-4 else 'FAIL'}")

rule("TEST test_int8_matches_training_off_ties")
print(f"frame {t}, first 6 dims:")
print(f"  CMVN'd x 32 (unrounded):  {x[t, :6]}")
print(f"  int8, training:           {want_q[t, :6]}")
print(f"  int8, runtime:            {q[t, :6]}")
off = np.argwhere(q != want_q)
print(f"values that differ: {len(off)} of {q.size}")
for f, k in off:
    print(
        f"  frame {f}, dim {k}: unrounded {x[f, k]:.6f} (|.5 - frac| = "
        f"{abs(abs(x[f, k]) % 1 - 0.5):.1e}), training {want_q[f, k]}, runtime {q[f, k]}"
    )
clipped = np.mean(np.abs(want_q.astype(int)) >= 127)
print(f"int8 range used: {want_q.min()}..{want_q.max()}, at the clamp: {clipped:.2%}")

# ---------------------------------------------------------------- figure
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
seq = LinearSegmentedColormap.from_list(
    "seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
div = LinearSegmentedColormap.from_list(
    "div", ["#1c5cab", "#86b6ef", "#f0efec", "#ec9a99", "#b42c2c"]
)
plt.rcParams.update(
    {
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": "#d6d5d0",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    }
)

T = len(feats)
fig, axes = plt.subplots(
    4,
    1,
    figsize=(11, 11),
    constrained_layout=True,
    gridspec_kw={"height_ratios": [1, 1.6, 2.4, 1.6]},
)

ax = axes[0]
ax.plot(np.arange(len(wav)) / sr * 100, wav, color="#2a78d6", lw=0.6)
ax.set_title("Input: waveform, 16 kHz (x axis in frames, 100 per second)", loc="left")
ax.set_xlim(0, T)
ax.set_ylabel("amplitude")

ax = axes[1]
im = ax.imshow(want_f[:, :41].T, aspect="auto", origin="lower", cmap=seq, interpolation="nearest")
ax.set_title(
    "Training's features, static part: 40 log-mel bands + energy (before CMVN)", loc="left"
)
ax.set_ylabel("dim (0 = lowest band)")
fig.colorbar(im, ax=ax, label="log power")

ax = axes[2]
im = ax.imshow(
    want_q.T.astype(int),
    aspect="auto",
    origin="lower",
    cmap=div,
    norm=TwoSlopeNorm(0, -128, 127),
    interpolation="nearest",
)
for y in (40.5, 81.5):
    ax.axhline(y, color=INK2, lw=0.6)
for (
    name,
    s,
) in GROUPS.items():
    if s.stop - s.start > 1:
        ax.text(T + 2, (s.start + s.stop) / 2, name.split()[0], va="center", fontsize=8, color=INK2)
ax.set_title(
    "Output: the int8 frames the fabric gets, all 123 dims (value / 32 = std devs)", loc="left"
)
ax.set_ylabel("dim")
fig.colorbar(im, ax=ax, label="int8")

ax = axes[3]
im = ax.imshow(d.T, aspect="auto", origin="lower", cmap=seq, interpolation="nearest")
for f, k in off:
    ax.plot(f, k, "o", ms=10, mfc="none", mec="#d03b3b", mew=1.5)
    ax.annotate(
        f"int8 differs by 1 (tie)\nframe {f}, dim {k}",
        (f, k),
        (12, 8),
        textcoords="offset points",
        fontsize=8,
        color=INK,
    )
ax.set_title(f"Check: |runtime - training| features (max {d.max():.1e}, limit 1e-4)", loc="left")
ax.set_xlabel("frame (10 ms each)")
ax.set_ylabel("dim")
fig.colorbar(im, ax=ax, label="|diff|")

out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).with_suffix(".png")
fig.savefig(out, dpi=130)
rule("FIGURE")
print(out)
