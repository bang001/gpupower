# A100 FP16 Tensor Core Energy Report

Date: 2026-06-03  
GPU: NVIDIA A100-SXM4-80GB, GA100, `sm_80`  
Canonical result directories:

- `../../../fp16_energy_impl/results/fp16_full_sweep_a100_20260603_044813/`
- `../../../fp16_energy_impl/results/fp16_matmul_pjbit_a100_20260603_031033/`
- `../../../fp16_energy_impl/results/fp16_long_sweep_a100_20260603_034822/`

## Summary

The recommended A100 diagnostic value is the full planned 5-second launch-shape sweep selected point:

| Basis | Launch shape | Repeats | Result |
|---|---|---:|---:|
| Quality-gate selected saturation point | `threads=384`, `blocks/SM=4` | 5 | `0.1144 +/- 0.0047 pJ/bit` |
| Lowest mean diagnostic point | `threads=192`, `blocks/SM=8` | 5 | `0.1050 +/- 0.0013 pJ/bit` |
| Selected target work-slope | `threads=384`, `blocks/SM=4` | 5 work points | `0.1406 pJ/bit`, R2 `0.986` |
| Lowest mean work-slope | `threads=192`, `blocks/SM=8` | 5 work points | `0.1337 pJ/bit`, R2 `0.990` |
| Re-measured old fixed launch | `threads=256`, `blocks/SM=8` | 5 | `0.1099 +/- 0.0073 pJ/bit` |
| Original short fixed run | `threads=256`, `blocks/SM=8` | last 10 of 12 | `0.1469 +/- 0.0109 pJ/bit` |

The `0.1144 +/- 0.0047 pJ/bit` value is preferred because it comes from the full planned launch-shape sweep and uses the quality-gate first-saturation selection rule. The lower `t192_b8` point is kept as a diagnostic lower-mean point, but it is not the selected representative target.

![A100 full 5s sweep](../../../fp16_energy_impl/results/fp16_full_sweep_a100_20260603_044813/figures/a100_long5s_sweep_pjbit_elapsed.png)

## What Is Being Measured

The quantity is not DRAM pJ/bit. It is a logical Tensor Core operand denominator:

```text
pJ/bit = (E_test - P_baseline * elapsed_test) / logical_FP16_input_bits
```

For one logical `m16n16k16` MMA:

```text
A bits + B bits = (16*16 + 16*16) * 16 = 8192 bit
FLOPs          = 2 * 16 * 16 * 16       = 8192 FLOP
```

The best name for the result is:

```text
logical m16n16k16 FP16 input bit당 baseline-subtracted board/NVML energy estimate
```

![FP16 experiment flow](../../../fp16_energy_impl/results/fp16_matmul_pjbit_a100_20260603_031033/figures/fp16_pjbit_experiment_flow.png)

## Baseline Subtraction

The test kernel is `tensor_mma_f16acc`. The structural baseline is `tensor_baseline_mov`. The baseline keeps the launch shape and register/synchronization structure close to the Tensor Core kernel while avoiding HMMA execution.

| Kernel | Role |
|---|---|
| `tensor_mma_f16acc` | FP16 Tensor Core HMMA workload |
| `tensor_baseline_mov` | structural no-HMMA baseline |

The subtraction removes much of the board/runtime/control overhead, but it does not isolate pure silicon arithmetic-unit energy.

![Baseline subtraction](../../../fp16_energy_impl/results/fp16_matmul_pjbit_a100_20260603_031033/figures/fp16_energy_subtraction_concept.png)

## Why The 5-Second Sweep Replaced The 1,000,000-Iteration Fixed Run

The original fixed run used `iters=1,000,000`, `threads=256`, `blocks/SM=8`, and `unroll=8`. It was useful for proving that the A100 binary and NVML measurement path worked, but the windows were short:

| Metric | Original fixed run | Full 5s sweep |
|---|---:|---:|
| Test elapsed | about `1.47 s` | new `full5s` minimum `5.891 s` |
| Baseline elapsed | about `0.43 s` | new `full5s` minimum `5.461 s` |
| Minimum test power samples | `10` | at least tens of samples per row |
| Same `t256_b8` pJ/bit | `0.1469 +/- 0.0109` | `0.1099 +/- 0.0073` |

The shorter fixed run is therefore retained as a historical diagnostic, but the full 5-second sweep is the preferred comparison basis.

## 5-Second Sweep Design

| Field | Setting |
|---|---|
| `threads/block` | `32`, `64`, `96`, `128`, `160`, `192`, `224`, `256`, `288`, `320`, `384` |
| `blocks/SM` | `1`, `2`, `4`, `8` |
| total launch shapes | 44 |
| `unroll` | 8 |
| `warmup` | 2 |
| baseline repeats | 10 |
| output store | suppressed |
| sample interval | 100 ms |
| primary energy source | `NVML total energy counter` |

The sweep scales `iters` so each test interval exceeds 5 seconds. The full planned range has complete 44/44 launch-shape coverage. Decision-boundary candidates `t160_b8`, `t192_b4`, `t192_b8`, `t224_b8`, `t288_b8`, `t320_b8`, and `t384_b2`, plus existing `t256_b8`, `t384_b4`, and `t384_b8`, were repeated to five runs for confidence intervals around the selected target and lower-mean candidates.

## Full Planned Sweep Range

```text
threads/block: 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384
blocks/SM:     1, 2, 4, 8
threads/SM:    32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384,
               512, 576, 640, 768, 896, 1024, 1152, 1280, 1536, 1792, 2048, 2304, 2560, 3072
logical shape: m16n16k16
input bits:    8192 bit/logical MMA
FLOPs:         8192 FLOP/logical MMA
```

The previous focused `3 x 4 = 12` A100 long-window run remains provenance for the first long-window result, but it is superseded for A100 launch-shape coverage by `fp16_full_sweep_a100_20260603_044813`.

## Selected Sweep Results

| Launch | Repeats | Test s | Baseline s | TFLOPS | Tensor model util | pJ/bit | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| `t192_b8` | 5 | 6.118 | 5.822 | 296.19 | 94.97% | `0.1050 +/- 0.0013` | lowest mean diagnostic |
| `t320_b8` | 5 | 5.891 | 5.461 | 307.57 | 98.62% | `0.1065 +/- 0.0052` | lower-mean repeated candidate |
| `t192_b4` | 5 | 6.364 | 6.139 | 284.73 | 91.30% | `0.1069 +/- 0.0051` | lower-mean repeated candidate |
| `t384_b8` | 5 | 5.879 | 5.355 | 308.18 | 98.82% | `0.1084 +/- 0.0054` | lower-mean repeated candidate |
| `t256_b8` | 5 | 5.897 | 5.706 | 307.26 | 98.52% | `0.1099 +/- 0.0073` | re-measured old launch |
| `t384_b4` | 5 | 5.884 | 5.628 | 307.95 | 98.74% | `0.1144 +/- 0.0047` | quality-gate selected |

The quality gate selects `t384_b4`, not the absolute lowest pJ/bit point, because the selection rule is the first Tensor Core model-utilization saturation point among valid candidates.

## Work-Slope Checks

| Basis | Launch | Work points | Slope | R2 | Status |
|---|---|---:|---:|---:|---|
| Selected target | `t384_b4` | 5 | `0.1406 pJ/bit` | `0.986` | positive slope |
| Lowest mean diagnostic | `t192_b8` | 5 | `0.1337 pJ/bit` | `0.990` | positive slope |

These work-slope sweeps varied `unroll = 1, 2, 4, 8, 16` at fixed `iters=2,000,000`, using the same `tensor_mma_f16acc` / `tensor_baseline_mov` pair. They are a consistency check that incremental energy increases with logical work; they do not replace the launch-shape sweep values above.

## Audit Summary

| Check | Status |
|---|---|
| A100 `sm_80` build/run | PASS |
| full planned sweep coverage | PASS, 44/44 launch shapes |
| full sweep timing | PASS, new `full5s` tests minimum `5.891 s` |
| decision-boundary repeats | PASS, selected/lower-mean candidates at 5 runs |
| current benchmark schema | PASS, `fp16-energy-bench-v2` |
| denominator metadata | PASS, `8192` input bits/logical MMA |
| intended timed global/L2 memory metadata | PASS, no intended memory |
| NVML total energy counter | PASS |
| clock stability | PASS, `1410 MHz`, span `0 MHz` |
| selected/lowest work-slope | PASS, positive slope and R2 >= 0.80 |
| NCU hardware counter validation | BLOCKED by `ERR_NVGPUCTRPERM` |

## NCU / Vast.ai Limitation

Root/sudo execution still failed with:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

The Vast.ai container exposes `RmProfilingAdminOnly: 1` and lacks the capability needed for GPU performance counters. Therefore HMMA activity, L2 bytes, DRAM bytes, and local spill counters are not validated. This remains a diagnostic NVML-counter estimate.

## Reproduction Artifacts

| Artifact | Path |
|---|---|
| full-sweep report source | `../../../fp16_energy_impl/results/fp16_full_sweep_a100_20260603_044813/` |
| full-sweep summary | `../../../fp16_energy_impl/results/fp16_full_sweep_a100_20260603_044813/thread_sweep_summary.csv` |
| full-sweep quality gate summary | `../../../fp16_energy_impl/results/fp16_full_sweep_a100_20260603_044813/quality_gate_summary.json` |
| selected target work-slope | `../../../fp16_energy_impl/results/fp16_full_sweep_a100_20260603_044813/work_slope_selected/work_slope_summary.csv` |
| lowest mean work-slope | `../../../fp16_energy_impl/results/fp16_full_sweep_a100_20260603_044813/work_slope_lowest_t192_b8/work_slope_summary.csv` |
| fixed-run report source | `../../../fp16_energy_impl/results/fp16_matmul_pjbit_a100_20260603_031033/` |
| prior focused long-sweep source | `../../../fp16_energy_impl/results/fp16_long_sweep_a100_20260603_034822/` |
