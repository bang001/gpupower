# A100 FP16 Long 5s Sweep Result Artifact

This file is kept as provenance for the raw result directory.

Canonical merged report:

```text
../../../docs/experiments/fp16_energy/A100_FP16_ENERGY_REPORT.md
```

Key result:

| Basis | Launch shape | Repeats | pJ/bit |
|---|---|---:|---:|
| quality-gate selected | `threads=384`, `blocks/SM=4` | 5 | `0.1144 +/- 0.0047` |
| lowest mean diagnostic | `threads=384`, `blocks/SM=8` | 5 | `0.1084 +/- 0.0054` |

Important artifacts in this directory:

- `thread_sweep_summary.csv`
- `quality_gate_summary.json`
- `summary.csv`
- `runs.jsonl`
- `figures/a100_long5s_sweep_pjbit_elapsed.png`
- `raw/`

The result remains diagnostic because NCU hardware counters are unavailable in the current Vast.ai environment.
