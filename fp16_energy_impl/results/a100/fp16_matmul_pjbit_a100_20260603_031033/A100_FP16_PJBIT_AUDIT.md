# A100 FP16 Fixed-Run Audit Artifact

This file is kept as provenance for the original A100 fixed-condition audit.

Canonical merged report and audit summary:

```text
../../../docs/experiments/fp16_energy/A100_FP16_ENERGY_REPORT.md
```

Audit conclusion:

| Check | Status |
|---|---|
| A100 `sm_80` build/run | PASS |
| raw run completeness | PASS |
| schema and denominator metadata | PASS |
| NVML total energy counter | PASS |
| clock stability | PASS |
| NCU hardware-counter validation | BLOCKED by `ERR_NVGPUCTRPERM` |

The detailed raw evidence remains in this result directory, especially `quality_gate_summary.json`, `summary.csv`, and `ncu_permission_probe*/`.
