#!/usr/bin/env bash
# Nsight Compute validation for dram_energy_decomp_cupy.py.
# This validates physical DRAM/L2/LSU counters separately from the NVML power run.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./run_energy_decomp_ncu.sh --device 0 --tag h100_decomp_ncu --dram-buf-bytes 8589934592

Options:
  --device N              CUDA/NVML GPU index. Default: 0
  --tag TAG               Output tag. Default: energy_decomp_ncu
  --stages "..."          Space-separated stages. Default: control_l2 l2 control_dram dram
  --dram-buf-bytes N      Passed to dram_energy_decomp_cupy.py
  --l2-buf-bytes N        Passed to dram_energy_decomp_cupy.py
  --l2-cache-op OP        cg or cs. Default: cg
  --dram-cache-op OP      cg or cs. Default: cs
  --ncu-bin PATH          Nsight Compute CLI. Default: ncu
  --ncu-metrics CSV       Explicit metric CSV, "auto", or "set". Default: auto
  --ncu-set NAME          Metric set fallback. Default: full
  --launch-skip N         Kernel launches to skip. Default: 2
  --launch-count N        Kernel launches to profile. Default: 1
  --out-dir DIR           Output directory. Default: reports
  --flat-output           Write directly to --out-dir
  --                       Extra args passed to run_energy_decomp.sh
EOF
}

DEVICE="0"
TAG="energy_decomp_ncu"
STAGES_STR="control_l2 l2 control_dram dram"
DRAM_BUF_BYTES=""
L2_BUF_BYTES=""
L2_CACHE_OP="cg"
DRAM_CACHE_OP="cs"
NCU_BIN="${NCU_BIN:-ncu}"
NCU_METRICS="auto"
NCU_SET="full"
LAUNCH_SKIP="2"
LAUNCH_COUNT="1"
OUT_DIR="reports"
FLAT_OUTPUT="0"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --stages) STAGES_STR="$2"; shift 2 ;;
    --dram-buf-bytes) DRAM_BUF_BYTES="$2"; shift 2 ;;
    --l2-buf-bytes) L2_BUF_BYTES="$2"; shift 2 ;;
    --l2-cache-op) L2_CACHE_OP="$2"; shift 2 ;;
    --dram-cache-op) DRAM_CACHE_OP="$2"; shift 2 ;;
    --ncu-bin) NCU_BIN="$2"; shift 2 ;;
    --ncu-metrics) NCU_METRICS="$2"; shift 2 ;;
    --ncu-set) NCU_SET="$2"; shift 2 ;;
    --launch-skip) LAUNCH_SKIP="$2"; shift 2 ;;
    --launch-count) LAUNCH_COUNT="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --flat-output) FLAT_OUTPUT="1"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) echo "[err] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

resolve_ncu_bin() {
  local requested="$1"
  local found
  if [[ "$requested" == */* ]]; then
    [[ -x "$requested" ]] && printf '%s' "$requested" && return 0
    return 1
  fi
  found="$(command -v "$requested" 2>/dev/null || true)"
  if [[ -n "$found" && -x "$found" ]]; then
    printf '%s' "$found"; return 0
  fi
  local old_nullglob
  old_nullglob="$(shopt -p nullglob || true)"
  shopt -s nullglob
  local candidates=(
    /usr/local/cuda/bin/ncu
    /usr/local/cuda-*/bin/ncu
    /usr/local/cuda/nsight-compute-*/ncu
    /usr/local/cuda-*/nsight-compute-*/ncu
    /opt/nvidia/nsight-compute/ncu
    /opt/nvidia/nsight-compute/*/ncu
  )
  eval "$old_nullglob"
  for found in "${candidates[@]}"; do
    [[ -x "$found" ]] && printf '%s' "$found" && return 0
  done
  return 1
}

metric_is_available() {
  local metric="$1"
  local escaped="${metric//./\\.}"
  grep -Eq "(^|[[:space:]])${escaped}([[:space:]]|$)" <<< "$AVAILABLE_NCU_METRICS"
}

join_by_comma() {
  local IFS=,
  printf '%s' "$*"
}

select_auto_ncu_metrics() {
  local selected=()
  local candidates=(
    gpu__time_duration.sum
    dram__bytes_read.sum
    dram__bytes_write.sum
    dram__sectors_read.sum
    dram__sectors_write.sum
    lts__t_sectors_op_read.sum
    lts__t_sectors_op_write.sum
    lts__t_sectors_srcunit_tex_op_read.sum
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
    smsp__inst_executed_pipe_lsu.sum
    smsp__inst_executed_pipe_fma.sum
    smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
    smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
  )
  for metric in "${candidates[@]}"; do
    if metric_is_available "$metric"; then
      selected+=("$metric")
    fi
  done
  join_by_comma "${selected[@]}"
}

kernel_regex_for_stage() {
  case "$1" in
    control_l2|control_dram) printf '%s' 'decomp_control_read' ;;
    l2|dram) printf '%s' 'decomp_stream_read_.*' ;;
    compute) printf '%s' 'decomp_compute_fma' ;;
    *) echo "[err] unknown stage: $1" >&2; exit 2 ;;
  esac
}

if ! NCU_BIN_RESOLVED="$(resolve_ncu_bin "$NCU_BIN")"; then
  echo "[err] Nsight Compute CLI not found: $NCU_BIN" >&2
  exit 1
fi
NCU_BIN="$NCU_BIN_RESOLVED"

mkdir -p "$OUT_DIR"
NCU_METRIC_ARGS=()
if [[ "$NCU_METRICS" == "auto" ]]; then
  AVAILABLE_NCU_METRICS="$("$NCU_BIN" --query-metrics --query-metrics-mode all 2>/dev/null || true)"
  AUTO_NCU_METRICS="$(select_auto_ncu_metrics)"
  if [[ -n "$AUTO_NCU_METRICS" ]]; then
    NCU_METRIC_ARGS=(--metrics "$AUTO_NCU_METRICS")
    echo "[info] ncu-metrics=$AUTO_NCU_METRICS"
  else
    NCU_METRIC_ARGS=(--set "$NCU_SET")
    echo "[warn] no known auto NCU metrics found; falling back to --set $NCU_SET" >&2
  fi
elif [[ "$NCU_METRICS" == "set" ]]; then
  NCU_METRIC_ARGS=(--set "$NCU_SET")
else
  NCU_METRIC_ARGS=(--metrics "$NCU_METRICS")
fi

read -r -a STAGES <<< "$STAGES_STR"
for stage in "${STAGES[@]}"; do
  kernel="$(kernel_regex_for_stage "$stage")"
  report_base="$OUT_DIR/${TAG}_${stage}"
  app_cmd=(
    "$SCRIPT_DIR/run_energy_decomp.sh"
    --device "$DEVICE"
    --only-stage "$stage"
    --targets 100
    --repeats 1
    --warmup-repeats 0
    --phase-seconds 1
    --idle-seconds 0.2
    --l2-cache-op "$L2_CACHE_OP"
    --dram-cache-op "$DRAM_CACHE_OP"
    --tag "${TAG}_${stage}"
    --out-dir "$OUT_DIR"
  )
  if [[ "$FLAT_OUTPUT" == "1" ]]; then
    app_cmd+=(--flat-output)
  fi
  if [[ -n "$DRAM_BUF_BYTES" ]]; then
    app_cmd+=(--dram-buf-bytes "$DRAM_BUF_BYTES")
  fi
  if [[ -n "$L2_BUF_BYTES" ]]; then
    app_cmd+=(--l2-buf-bytes "$L2_BUF_BYTES")
  fi
  app_cmd+=("${EXTRA_ARGS[@]}")

  echo
  echo "[ncu] stage=$stage kernel=$kernel report=${report_base}.ncu-rep"
  set +e
  "$NCU_BIN" \
    --target-processes all \
    --kernel-name "regex:${kernel}" \
    --launch-skip "$LAUNCH_SKIP" \
    --launch-count "$LAUNCH_COUNT" \
    --force-overwrite \
    --export "$report_base" \
    "${NCU_METRIC_ARGS[@]}" \
    "${app_cmd[@]}" 2>&1 | tee "${report_base}.ncu.log"
  status=${PIPESTATUS[0]}
  set -e
  if grep -q "ERR_NVGPUCTRPERM" "${report_base}.ncu.log"; then
    echo "[err] NCU performance counter permission denied." >&2
    echo "      Run sudo -v first or ask admin to set NVreg_RestrictProfilingToAdminUsers=0." >&2
    exit 13
  fi
  if [[ "$status" -ne 0 ]]; then
    echo "[err] NCU failed with exit code $status; log=${report_base}.ncu.log" >&2
    exit "$status"
  fi
done

echo
echo "[done] energy-decomposition NCU reports: $OUT_DIR/${TAG}_*.ncu-rep"
