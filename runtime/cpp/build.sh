#!/usr/bin/env bash
# Build the fabric's C++ job loop into src/runtime/ as `runtime._jobs`. Board only (it
# drives the fabric's registers), with the Python that will import it, PYNQ's venv's, which
# has pybind11. A few seconds on the board.
#
#     PYTHON=/usr/local/share/pynq-venv/bin/python3 build.sh
set -euo pipefail
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")"
OUT=../src/runtime/_jobs$("$PYTHON" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
g++ -std=c++17 -O2 -Wall -Wextra -fPIC -shared $("$PYTHON" -m pybind11 --includes) jobs.cpp -o "$OUT"
echo "built $(realpath "$OUT")"
