# A100 FP16 Tensor Core Energy Report

Date: 2026-06-03  
GPU: NVIDIA A100-SXM4-80GB, GA100, `sm_80`  
Canonical result directories:

- `../../../fp16_energy_impl/results/fp16_matmul_pjbit_a100_20260603_031033/`
- `../../../fp16_energy_impl/results/fp16_long_sweep_a100_20260603_034822/`

## Summary

The recommended A100 diagnostic value is the 5-second launch-shape sweep selected point:

| Basis | Launch shape | Repeats | Result |
|---|---|---:|---:|
| Quality-gate selected saturation point | `threads=384`, `blocks/SM=4` | 5 | `0.1144 +/- 0.0047 pJ/bit` |
| Lowest mean diagnostic point | `threads=384`, `blocks/SM=8` | 5 | `0.1084 +/- 0.0054 pJ/bit` |
| Re-measured old fixed launch | `threads=256`, `blocks/SM=8` | 5 | `0.1099 +/- 0.0073 pJ/bit` |
| Original short fixed run | `threads=256`, `blocks/SM=8` | last 10 of 12 | `0.1469 +/- 0.0109 pJ/bit` |

The `0.1144 +/- 0.0047 pJ/bit` value is preferred because it comes from a launch-shape sweep and uses the quality-gate first-saturation selection rule. The lower `t384_b8` point is kept as a diagnostic lower-mean point, but it is not the selected representative target.

![A100 long 5s sweep](../../../fp16_energy_impl/results/fp16_long_sweep_a100_20260603_034822/figures/a100_long5s_sweep_pjbit_elapsed.png)

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

| Metric | Original fixed run | 5s long sweep |
|---|---:|---:|
| Test elapsed | about `1.47 s` | minimum `5.878 s` |
| Baseline elapsed | about `0.43 s` | minimum `5.353 s` |
| Minimum test power samples | `10` | `45` |
| Minimum baseline power samples | `3` | `41` |
| Same `t256_b8` pJ/bit | `0.1469 +/- 0.0109` | `0.1099 +/- 0.0073` |

The shorter fixed run is therefore retained as a historical diagnostic, but the 5-second sweep is the preferred comparison basis.

## 5-Second Sweep Design

| Field | Setting |
|---|---|
| `threads/block` | `128`, `256`, `384` |
| `blocks/SM` | `1`, `2`, `4`, `8` |
| total launch shapes | 12 |
| `unroll` | 8 |
| `warmup` | 2 |
| baseline repeats | 10 |
| output store | suppressed |
| sample interval | 100 ms |
| primary energy source | `NVML total energy counter` |

The sweep scales `iters` so each test interval is roughly 5.9 seconds. Decision points `t256_b8`, `t384_b4`, and `t384_b8` were repeated to five runs for a confidence interval.

## Selected Sweep Results

| Launch | Repeats | Test s | Baseline s | TFLOPS | Tensor model util | pJ/bit | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| `t384_b8` | 5 | 5.879 | 5.355 | 308.18 | 98.82% | `0.1084 +/- 0.0054` | lowest mean diagnostic |
| `t384_b2` | 1 | 5.892 | 5.770 | 307.51 | 98.60% | `0.1097` | single scan point |
| `t256_b8` | 5 | 5.897 | 5.706 | 307.26 | 98.52% | `0.1099 +/- 0.0073` | re-measured old launch |
| `t384_b4` | 5 | 5.884 | 5.628 | 307.95 | 98.74% | `0.1144 +/- 0.0047` | quality-gate selected |

The quality gate selects `t384_b4`, not the absolute lowest pJ/bit point, because the selection rule is the first Tensor Core model-utilization saturation point among valid candidates.

## Audit Summary

| Check | Status |
|---|---|
| A100 `sm_80` build/run | PASS |
| raw run completeness, fixed run | PASS, 48/48 raw runs |
| long sweep pair rows | PASS, 24 pair rows |
| current benchmark schema | PASS, `fp16-energy-bench-v2` |
| denominator metadata | PASS, `8192` input bits/logical MMA |
| intended timed global/L2 memory metadata | PASS, no intended memory |
| NVML total energy counter | PASS |
| clock stability | PASS, `1410 MHz`, span `0 MHz` |
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
| fixed-run report source | `../../../fp16_energy_impl/results/fp16_matmul_pjbit_a100_20260603_031033/` |
| long-sweep report source | `../../../fp16_energy_impl/results/fp16_long_sweep_a100_20260603_034822/` |
| long-sweep summary | `../../../fp16_energy_impl/results/fp16_long_sweep_a100_20260603_034822/thread_sweep_summary.csv` |
| quality gate summary | `../../../fp16_energy_impl/results/fp16_long_sweep_a100_20260603_034822/quality_gate_summary.json` |
