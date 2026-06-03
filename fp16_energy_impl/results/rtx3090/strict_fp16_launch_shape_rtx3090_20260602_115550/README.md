# RTX 3090 Strict-Like Diagnostic Artifact

This file is kept as provenance for the RTX3090 strict-like diagnostic run.

Canonical merged report:

```text
../../../docs/experiments/fp16_energy/RTX3090_FP16_RESULTS.md
```

Key result:

| Launch shape | TFLOPS | pJ/bit | Status |
|---|---:|---:|---|
| `threads=256`, `blocks/SM=1` | 159.148 | `0.3085 +/- 0.0253` | strict-like diagnostic; NCU blocked |

Important artifacts:

- `thread_sweep_summary.csv`
- `quality_gate_summary.json`
- `resource_audit/`
- `ncu_no_l2_thread_sweep/`
- `figures/`
