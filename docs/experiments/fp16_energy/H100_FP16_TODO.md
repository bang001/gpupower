# H100 FP16 Energy Experiment TODO

Date: 2026-06-03
Target GPU: NVIDIA H100, Hopper, expected `sm_90`
Output root: `../../../fp16_energy_impl/results/h100/`

This document tracks the planned H100 FP16 Tensor Core energy experiment. The goal is to produce a result that is directly comparable to the current A100 diagnostic value:

```text
logical m16n16k16 FP16 FLOP당 baseline-subtracted board/NVML energy estimate
```

The measurement is not DRAM pJ/bit and not pure Tensor Core silicon energy. It is a baseline-subtracted board/NVML estimate using the same logical `pJ/FLOP` denominator as the A100 and RTX3090 reports. Legacy generated fields may still mention input bits; for logical `m16n16k16`, the A/B input-bit denominator and FLOP denominator are both `8192` per logical MMA.

## Target Deliverables

| Deliverable | Planned path |
|---|---|
| H100 canonical report | `docs/experiments/fp16_energy/H100_FP16_ENERGY_REPORT.md` |
| H100 full sweep CSV | `docs/experiments/fp16_energy/H100_FP16_FULL_SWEEP_TABLE.csv` |
| H100 full sweep Excel | `docs/experiments/fp16_energy/H100_FP16_FULL_SWEEP_TABLE.xlsx` |
| Raw result directory | `fp16_energy_impl/results/h100/fp16_full_sweep_h100_<YYYYMMDD_HHMMSS>/` |
| Result-local report/table copies | Same raw result directory |

## Experiment Plan

| Step | Status | Notes |
|---|---|---|
| Reserve H100 environment | TODO | Record provider, instance type, GPU UUID, driver, CUDA, container image, and power policy. |
| Capture preflight | TODO | Run GPU/runtime preflight before compiling or benchmarking. Save JSON/CSV under the H100 result directory. |
| Build for Hopper | TODO | Compile the benchmark with `sm_90` support and save build logs, ptxas output, and binary hash. |
| Validate timer and energy source | TODO | Confirm CUDA event timing, NVML total energy counter availability, and power-sampling cadence. |
| Run smoke test | TODO | One short baseline/test pair for `tensor_mma_f16acc` versus `tensor_baseline_mov`; verify positive delta and sane TFLOPS. |
| Run full 5s launch-shape sweep | TODO | Use the same `11 x 4 = 44` launch-shape grid as A100 for cross-GPU comparability. |
| Repeat decision-boundary candidates | TODO | Repeat selected saturation and lower-mean candidates to 5 runs for CI. |
| Run work-slope checks | TODO | Use multiple work sizes for selected and lower-mean candidates; require positive slope and high R2. |
| Run NCU permission probe | TODO | Determine whether hardware counters are accessible. Document `ERR_NVGPUCTRPERM` if blocked. |
| Run strict NCU validation if possible | TODO | Validate HMMA activity, DRAM/L2 bytes, local memory, occupancy, and tensor utilization if counters are available. |
| Generate tables | TODO | Produce Markdown table, CSV, and XLSX with all launch shapes and selection status. |
| Update comparison docs | TODO | Add H100 to A100/RTX3090 comparison only after sweep, repeats, and claim boundary are documented. |

## Baseline Sweep Grid

Use the same launch-shape grid as the A100 full sweep first. Extend only if H100 saturation is not clearly reached.

```text
threads/block: 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384
blocks/SM:     1, 2, 4, 8
threads/SM:    32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384,
               512, 576, 640, 768, 896, 1024, 1152, 1280, 1536, 1792, 2048, 2304, 2560, 3072
logical shape: m16n16k16
input bits:    8192 bit/logical MMA
FLOPs:         8192 FLOP/logical MMA
```

Selection rule:

1. Filter rows that fail basic quality gates.
2. Prefer the first tensor-utilization saturation point among valid rows.
3. Report the absolute lower-mean point separately if it differs from the selected target.
4. Do not promote the lower-mean point to canonical representative value unless the selection rule is explicitly changed and documented.

## API 호출 관계 유의사항

H100 실험은 측정 중 불필요한 API 호출 노이즈를 줄이고, 외부 자동화 API의 rate limit 실패가 실험 재실행으로 이어지지 않도록 관리한다.

| Area | Risk | Required practice |
|---|---|---|
| CUDA runtime calls | Host-side API calls can perturb timing if placed inside the measurement path. | Keep CUDA setup outside the measured interval. Use CUDA events around the kernel window and synchronize once per measured pair. |
| Kernel launch API | Launch overhead can dominate short windows. | Keep test windows at >=5 seconds and save elapsed time for both baseline and test. |
| NVML API | Very frequent power queries add overhead and can return missing/quantized values. | Initialize NVML once, sample at the established cadence, store raw samples, and record whether total energy counter or integrated power was used. |
| NVML energy counter | Counter support and resolution may differ on H100 environments. | Probe support explicitly. If unsupported, mark the run as power-integration based and do not compare it as equivalent without noting the source. |
| Nsight Compute / CUPTI | Hardware counters may be blocked by admin policy, producing `ERR_NVGPUCTRPERM`. | Run a permission probe before strict validation. Do not repeatedly retry blocked NCU runs. Document the exact error and environment capability state. |
| Profiling serialization | Concurrent profiling or overlapping benchmark processes can invalidate results. | Run one benchmark/profiling job at a time. Record process start/end times and GPU UUID. |
| Cloud/provider API | Instance labels may not uniquely identify H100 SKU or clock/power policy. | Capture `nvidia-smi -q`, GPU UUID, power limit, clocks, MIG state, and persistence mode in the result directory. |
| Assistant/OpenAI API usage | Repeated status polling or retries can hit `429 Too Many Requests` and does not advance the local benchmark. | Start long local experiments as durable commands, save logs/checkpoints locally, and ask for status by inspecting local files/processes rather than reissuing experiment commands. |
| Retry behavior | Retrying a full sweep after an API failure can duplicate artifacts or bias summaries. | Make commands idempotent with unique output directories and manifests. Resume from recorded state when possible. |
| Secrets and tokens | API keys can leak into logs, reports, or result bundles. | Never write provider tokens, OpenAI keys, GitHub tokens, or SSH material into `results/` or docs. Redact environment captures if needed. |

## Minimum Preflight Checklist

| Check | Required record |
|---|---|
| GPU identity | GPU name, UUID, PCIe/SXM form if known, SM count, MIG state |
| Software | Driver, CUDA toolkit, runtime CUDA, compiler, Python, benchmark git commit |
| Power/clock policy | Power limit, application clocks if set, persistence mode, throttling reasons |
| Permissions | NCU permission probe result and any `ERR_NVGPUCTRPERM` text |
| Build | `sm_90` build log, ptxas resource output, binary hash |
| Measurement source | NVML energy counter support, power sample interval, fallback policy |
| Interference | Other GPU processes before and during run |

## Definition Of Done

The H100 result is ready to report when all of the following are true:

| Requirement | Threshold |
|---|---|
| Full sweep coverage | 44/44 planned launch shapes completed |
| Test window | Every selected/reportable row uses >=5 seconds test elapsed |
| Repeats | Selected target and lower-mean candidate have 5 repeats |
| Work slope | Selected target has positive slope and acceptable R2 |
| Artifacts | Raw CSV/JSON/logs, report, CSV table, and XLSX table are committed |
| Claim boundary | NVML-only versus NCU-validated status is stated explicitly |
| Comparison update | H100 is added to comparison docs only after the above gates pass |

## Open Questions Before Running

| Question | Why it matters |
|---|---|
| H100 SXM or PCIe? | Power limits, clocks, and cooling can change board-energy estimates. |
| NCU counters available? | Determines whether the H100 result can move beyond diagnostic NVML estimate. |
| MIG enabled? | MIG can change visible SMs, clocks, and counter availability. |
| Fixed clocks possible? | Fixed clocks improve cross-run stability, but may require admin privileges. |
| Same container/toolchain as A100? | Reduces software drift in cross-GPU comparison. |
