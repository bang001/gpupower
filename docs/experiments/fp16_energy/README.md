# FP16 Energy Experiment Documentation

This directory is the canonical documentation location for the FP16 Tensor Core pJ/bit experiments. The `fp16_energy_impl/results/` tree is kept as raw provenance: CSV, JSON, power traces, generated figures, and per-run metadata.

## Canonical Documents

| Document | Purpose |
|---|---|
| [A100_FP16_ENERGY_REPORT.md](A100_FP16_ENERGY_REPORT.md) | A100 GA100 FP16 Tensor Core report, including fixed run, 5s sweep, audit, and NCU limitation |
| [RTX3090_FP16_RESULTS.md](RTX3090_FP16_RESULTS.md) | RTX 3090 GA102 strict-like and diagnostic result history |
| [FP16_A100_RTX3090_COMPARISON.md](FP16_A100_RTX3090_COMPARISON.md) | Cross-GPU comparison and claim boundary |

## Current Recommended Diagnostic Values

| GPU | Result | Launch shape | Status |
|---|---:|---|---|
| A100-SXM4-80GB | `0.1144 +/- 0.0047 pJ/bit` | `threads=384`, `blocks/SM=4` | quality-gate selected 5s diagnostic |
| A100-SXM4-80GB lowest mean | `0.1084 +/- 0.0054 pJ/bit` | `threads=384`, `blocks/SM=8` | lower pJ/bit, not first-saturation selected target |
| RTX 3090 | `0.3085 +/- 0.0253 pJ/bit` | `threads=256`, `blocks/SM=1` | strict-like diagnostic, NCU blocked |

![Updated A100 RTX3090 comparison](images/fp16_a100_rtx3090_updated_comparison.png)

## Claim Boundary

All current values are diagnostic estimates, not final strict claims. They use `NVML total energy counter` deltas and structural baseline subtraction. Nsight Compute hardware counters are blocked by `ERR_NVGPUCTRPERM`, so HMMA activity, L2/DRAM bytes, and local spill counters are not independently validated.
