# FP16 Energy Result Artifacts

This directory stores raw experiment artifacts: CSV/JSON summaries, power traces, generated figures, NCU permission probes, and per-run metadata.

Canonical human-readable reports now live under:

```text
../../docs/experiments/fp16_energy/
```

## Canonical Reports

| Report | Purpose |
|---|---|
| [A100 FP16 energy report](../../docs/experiments/fp16_energy/A100_FP16_ENERGY_REPORT.md) | A100 fixed run, full 5s sweep, boundary repeats, work-slope checks, audit, and NCU limitation |
| [RTX 3090 FP16 results](../../docs/experiments/fp16_energy/RTX3090_FP16_RESULTS.md) | RTX3090 strict-like and diagnostic result history |
| [A100 vs RTX3090 comparison](../../docs/experiments/fp16_energy/FP16_A100_RTX3090_COMPARISON.md) | Cross-GPU comparison and claim boundary |

## Key Result Directories

| Directory | Role | Canonical value/status |
|---|---|---|
| `fp16_full_sweep_a100_20260603_044813/` | A100 full planned 5s launch-shape sweep plus boundary repeats/work-slope | selected `0.1144 +/- 0.0047 pJ/bit`; lowest mean `0.1050 +/- 0.0013 pJ/bit` |
| `fp16_long_sweep_a100_20260603_034822/` | A100 focused 5s launch-shape sweep | superseded by full planned sweep; selected `0.1144 +/- 0.0047 pJ/bit` |
| `fp16_matmul_pjbit_a100_20260603_031033/` | A100 original fixed run plus audit | historical fixed value `0.1469 +/- 0.0109 pJ/bit`; superseded for comparison by 5s sweep |
| `strict_fp16_launch_shape_rtx3090_20260602_115550/` | RTX3090 strict-like diagnostic | selected `0.3085 +/- 0.0253 pJ/bit`, NCU blocked |
| `diagnostic_fp16_launch_shape_rtx3090_20260602_125100/` | RTX3090 no-NCU diagnostic sweep | diagnostic only, no strict target |
| `strict_fp16_launch_shape_rtx3090_20260602_124900/` | RTX3090 strict pipeline attempt | stopped at NCU permission probe |

## Planned Sweep Range

The broader launch-shape range that should be retained for architecture-comparison runs is:

```text
threads/block: 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384
blocks/SM:     1, 2, 4, 8
threads/SM:    32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384,
               512, 576, 640, 768, 896, 1024, 1152, 1280, 1536, 1792, 2048, 2304, 2560, 3072
logical shape: m16n16k16
input bits:    8192 bit/logical MMA
FLOPs:         8192 FLOP/logical MMA
```

The current A100 result used the full `11 x 4 = 44` range with >=5-second test windows and boundary-repeat reinforcement. The current RTX 3090 strict-like diagnostic artifact used `threads/block = 32, 64, 128, 256` and `blocks/SM = 1, 2, 4, 8`. See the canonical [FP16 experiment documentation](../../docs/experiments/fp16_energy/README.md) for the selection rule and claim boundary.

## Claim Boundary

Current A100 and RTX3090 values are diagnostic NVML-counter estimates. Nsight Compute hardware-counter validation is blocked by `ERR_NVGPUCTRPERM` in the available environments, so these are not final strict publishable hardware-counter claims.
