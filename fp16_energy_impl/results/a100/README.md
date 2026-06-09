# A100 FP16 Energy Results

A100/GA100 result artifacts live here.

| Directory | Role | Status |
|---|---|---|
| `fp16_full_sweep_a100_20260603_044813/` | Full planned 5s launch-shape sweep, boundary repeats, work-slope checks | Current canonical A100 diagnostic result |
| `fp16_long_sweep_a100_20260603_034822/` | Focused 5s long sweep | Superseded by the full planned sweep |
| `fp16_matmul_pjbit_a100_20260603_031033/` | Original fixed run plus audit figures | Historical fixed-run provenance |

## Current Full Sweep Summary

| Item | Value |
|---|---|
| Launch-shape coverage | 44/44 planned shapes |
| Selected representative target | `threads=384`, `blocks/SM=4`, `0.1144 +/- 0.0047 pJ/FLOP` |
| Lowest repeated mean | `threads=192`, `blocks/SM=8`, `0.1050 +/- 0.0013 pJ/FLOP` |
| Selected work-slope check | `0.1406 pJ/FLOP`, `R2=0.9865` |
| Lowest-mean work-slope check | `0.1337 pJ/FLOP`, `R2=0.9900` |

## Tables

| Artifact | Path |
|---|---|
| Result-local CSV | `fp16_full_sweep_a100_20260603_044813/A100_FP16_FULL_SWEEP_TABLE.csv` |
| Result-local Excel | `fp16_full_sweep_a100_20260603_044813/A100_FP16_FULL_SWEEP_TABLE.xlsx` |
| Documentation CSV | `../../../docs/experiments/fp16_energy/A100_FP16_FULL_SWEEP_TABLE.csv` |
| Documentation Excel | `../../../docs/experiments/fp16_energy/A100_FP16_FULL_SWEEP_TABLE.xlsx` |
