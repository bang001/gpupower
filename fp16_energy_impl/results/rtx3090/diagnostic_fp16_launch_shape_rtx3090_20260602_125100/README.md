# RTX 3090 No-NCU Diagnostic Sweep Artifact

This file is kept as provenance for the diagnostic RTX3090 run.

Canonical merged report:

```text
../../../docs/experiments/fp16_energy/RTX3090_FP16_RESULTS.md
```

Key diagnostic point:

| Launch shape | TFLOPS | pJ/bit | Status |
|---|---:|---:|---|
| `threads=256`, `blocks/SM=1` | 159.825 | `0.1833 +/- 0.1084` | no strict selected target; NCU skipped/blocked |

Important artifacts:

- `thread_sweep_summary.csv`
- `quality_gate_summary.json`
- `quality_gates.csv`
- `resource_audit/`
- `figures/`
