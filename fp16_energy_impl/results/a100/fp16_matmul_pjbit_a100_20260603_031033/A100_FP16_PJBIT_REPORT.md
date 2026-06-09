# A100 FP16 Fixed-Run Result Artifact

This file is kept as provenance for the original A100 fixed-condition result directory.

Canonical merged report:

```text
../../../docs/experiments/fp16_energy/A100_FP16_ENERGY_REPORT.md
```

Original fixed-condition result:

| Launch shape | Repeats | pJ/FLOP |
|---|---:|---:|
| `threads=256`, `blocks/SM=8`, `iters=1,000,000` | last 10 of 12 | `0.1469 +/- 0.0109` |

The original generated field was `matmul_input_pj_per_bit`. For this logical `m16n16k16` Tensor Core workload, A/B input bits and FLOPs are both `8192` per logical MMA, so the same numeric value is reported as `pJ/FLOP`.

This value is now treated as a short-window historical diagnostic. The 5-second sweep result in `../fp16_long_sweep_a100_20260603_034822/` is the preferred A100 comparison basis.

Important artifacts in this directory:

- `A100_FP16_PJBIT_AUDIT.md`
- `summary.csv`
- `condition_summary.csv`
- `quality_gate_summary.json`
- `ncu_permission_probe*/`
- `figures/`
- `raw/`
