#!/usr/bin/env bash
# Build the C++ beam search into src/shared/decode/ as `shared.decode.beam_cpp`: KenLM's query
# sources (the list its old setup.py compiled, 4 at a time) and beam_cpp.cpp, linked into one
# extension. Needs g++, zlib, pybind11 in the Python that will import it, and KenLM's source
# (the kenlm-0.3.0 sdist the wlm extra builds from). About 2 minutes on the board.
#
#     PYTHON=<python> build.sh <kenlm source dir>     (PYTHON defaults to `python`)
set -euo pipefail
KENLM=$(realpath "$1")
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")"
mkdir -p build
FLAGS="-O3 -DNDEBUG -DKENLM_MAX_ORDER=6 -DHAVE_ZLIB -DHAVE_CLOCKGETTIME -fPIC -I$KENLM"
ls "$KENLM"/util/*.cc "$KENLM"/lm/*.cc "$KENLM"/util/double-conversion/*.cc \
  | grep -vE '(main|test)\.cc$' \
  | xargs -P4 -I{} sh -c "g++ -std=c++11 $FLAGS -c {} -o build/\$(basename \$(dirname {}))_\$(basename {} .cc).o"
OUT=../src/shared/decode/beam_cpp$("$PYTHON" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
# -ffp-contract=off: no fused multiply-adds, so the arithmetic rounds as Python's does.
g++ -std=c++17 -ffp-contract=off $FLAGS $("$PYTHON" -m pybind11 --includes) -shared beam_cpp.cpp \
  build/*.o -lz -lrt -o "$OUT"
echo "built $(realpath "$OUT")"
