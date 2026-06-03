# FP16 Energy Experiment Documentation

This directory is the canonical documentation location for the FP16 Tensor Core pJ/bit experiments. The `fp16_energy_impl/results/` tree is kept as raw provenance: CSV, JSON, power traces, generated figures, and per-run metadata.

## Canonical Documents

| Document | Purpose |
|---|---|
| [A100_FP16_ENERGY_REPORT.md](A100_FP16_ENERGY_REPORT.md) | A100 GA100 FP16 Tensor Core report, including full planned 5s sweep, boundary repeats, work-slope checks, audit, and NCU limitation |
| [RTX3090_FP16_RESULTS.md](RTX3090_FP16_RESULTS.md) | RTX 3090 GA102 strict-like and diagnostic result history |
| [FP16_A100_RTX3090_COMPARISON.md](FP16_A100_RTX3090_COMPARISON.md) | Cross-GPU comparison and claim boundary |

## Current Recommended Diagnostic Values

| GPU | Result | Launch shape | Status |
|---|---:|---|---|
| A100-SXM4-80GB | `0.1144 +/- 0.0047 pJ/bit` | `threads=384`, `blocks/SM=4` | quality-gate selected 5s diagnostic |
| A100-SXM4-80GB lowest mean | `0.1050 +/- 0.0013 pJ/bit` | `threads=192`, `blocks/SM=8` | lower pJ/bit, not first-saturation selected target |
| RTX 3090 | `0.3085 +/- 0.0253 pJ/bit` | `threads=256`, `blocks/SM=1` | strict-like diagnostic, NCU blocked |

![Updated A100 RTX3090 comparison](images/fp16_a100_rtx3090_updated_comparison.png)

## Launch-Shape Sweep Range

The broad launch-shape range planned for architecture-comparison runs is:

```text
threads/block: 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384
blocks/SM:     1, 2, 4, 8
threads/SM:    32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384,
               512, 576, 640, 768, 896, 1024, 1152, 1280, 1536, 1792, 2048, 2304, 2560, 3072
logical shape: m16n16k16
input bits:    8192 bit/logical MMA
FLOPs:         8192 FLOP/logical MMA
```

The current A100 follow-up completed this full `11 x 4 = 44` launch-shape range with >=5-second test windows, then repeated the selected and lower-mean boundary candidates to five runs. The RTX 3090 strict-like diagnostic run used `threads/block = 32, 64, 128, 256` and `blocks/SM = 1, 2, 4, 8`.

Target selection is not based on the lowest pJ/bit row alone. The current rule is to choose the first saturation point among rows that pass the quality gate and tensor-model-utilization sanity checks. This is why the A100 representative target is `threads=384`, `blocks/SM=4`, while `threads=192`, `blocks/SM=8` is reported separately as the lower-mean diagnostic point after boundary-repeat reinforcement.

## Claim Boundary

All current values are diagnostic estimates, not final strict claims. They use `NVML total energy counter` deltas and structural baseline subtraction. Nsight Compute hardware counters are blocked by `ERR_NVGPUCTRPERM`, so HMMA activity, L2/DRAM bytes, and local spill counters are not independently validated.
