# RTX 3090 FP16 Energy Results

RTX 3090/GA102 result artifacts live here, including strict-like, diagnostic, NCU-permission, and older fixed-run provenance.

| Directory/File | Role | Status |
|---|---|---|
| `strict_fp16_launch_shape_rtx3090_20260602_115550/` | Strict-like launch-shape diagnostic | Current RTX3090 diagnostic target `0.3085 +/- 0.0253 pJ/bit` at `threads=256`, `blocks/SM=1` |
| `diagnostic_fp16_launch_shape_rtx3090_20260602_125100/` | No-NCU diagnostic sweep | Diagnostic only |
| `strict_fp16_launch_shape_rtx3090_20260602_124900/` | Strict pipeline attempt | Stopped at NCU permission probe |
| `strict_fp16_launch_shape_rtx3090_ncu_20260602_194341/` | NCU permission probe rerun | Permission-limited provenance |
| `fp16_work_slope_bar_repeat30_rtx3090_20260601/` | Work-slope repeat history | Historical provenance |
| `fp16_matmul_thread_sweep_fine_m16n16_smutil_rtx3090_20260528/` | Earlier thread sweep | Historical provenance |
| `fp16_matmul_pjbit_gpu0_default_clock_cuda132_20260528_1331/` | Earlier fixed run | Historical provenance |
| `architecture_compare_rtx3090_readiness_20260602/` | Architecture-readiness notes/artifacts | Historical provenance |
| `ncu_gpu0_default_clock_cuda132/` | Earlier NCU text artifact | Historical provenance |
| `env_gpu0_cuda132_conda_20260528_1335.txt` | Environment capture | Historical provenance |
