#!/usr/bin/env bash
# Write-only SM/L2/DRAM energy decomposition launcher.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PY:-}"
if [[ -z "$PY" ]]; then
    for cand in \
        "${VIRTUAL_ENV:-}/bin/python" \
        "${CONDA_PREFIX:-}/bin/python" \
        /home/bang001/miniforge3/envs/ssc21env/bin/python \
        "$SCRIPT_DIR/.venv-pjbit/bin/python" \
        "$(command -v python3 || true)"; do
        if [[ -x "$cand" ]] && "$cand" -c "import cupy, pynvml, matplotlib" >/dev/null 2>&1; then
            PY="$cand"
            break
        fi
    done
elif ! "$PY" -c "import cupy, pynvml, matplotlib" >/dev/null 2>&1; then
    echo "[err] PY does not provide cupy/pynvml/matplotlib: $PY" >&2
    echo "      If using sudo with a venv, run:" >&2
    echo "      sudo env PY=/path/to/venv/bin/python ./run_write_energy_decomp.sh ..." >&2
    exit 1
fi

if [[ -z "$PY" ]]; then
    echo "[err] cupy/pynvml/matplotlib 환경 필요" >&2
    echo "      pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12 pynvml matplotlib" >&2
    echo "      sudo는 venv 환경을 지울 수 있으므로 필요하면 다음처럼 실행:" >&2
    echo "      sudo env PY=/path/to/venv/bin/python ./run_write_energy_decomp.sh ..." >&2
    exit 1
fi

# Defaults are intentionally report-grade rather than smoke-test-grade:
# 35 repeats, first 5 repeats discarded, median aggregation, 1s duty window.
# Override on the command line for quick testing, e.g. --repeats 7 --warmup-repeats 2.
exec "$PY" "$SCRIPT_DIR/dram_write_energy_decomp_cupy.py" \
  --targets 50 75 100 \
  --repeats 35 \
  --warmup-repeats 5 \
  --phase-seconds 15 \
  --idle-seconds 20 \
  --window-ms 1000 \
  --gap-seconds 1 \
  --fit-aggregate median \
  --write-pattern address \
  --l2-cache-op wb \
  --dram-cache-op cs \
  "$@"
