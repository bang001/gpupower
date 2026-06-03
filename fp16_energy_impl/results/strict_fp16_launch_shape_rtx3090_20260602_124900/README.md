# RTX 3090 Strict Pipeline Attempt Artifact

This file is kept as provenance for the strict pipeline attempt that stopped at NCU permission probing.

Canonical merged report:

```text
../../../docs/experiments/fp16_energy/RTX3090_FP16_RESULTS.md
```

Status:

| Check | Result |
|---|---|
| GPU/runtime preflight | pass |
| Architecture check | pass, RTX3090/GA102, CUDA arch 86 |
| Build | pass |
| NCU permission probe | fail, `ERR_NVGPUCTRPERM` |
| Strict sweep | not started |

Important artifact:

```text
ncu_permission_probe/ncu_permission_probe.json
```
