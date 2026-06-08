# A100 FP16 Operand Variation 진단 실험

날짜: 2026-06-08
GPU: NVIDIA A100-SXM4-80GB
Launch shape: `threads=384`, `blocks/SM=4`
측정 상태: NCU 사용 불가, NVML-only diagnostic

## 요약

| 조건 | Runs | Valid no-L2 | TFLOPS mean | pJ/FLOP mean | CI95 |
|---|---:|---:|---:|---:|---:|
| Fixed control | 5 | 5 | `306.49` | `0.1204` | `0.0066` |
| Warp-static 4-set | 5 | 5 | `310.49` | `0.2575` | `0.0029` |
| Warp-rotating 4-set | 5 | 5 | `292.52` | `0.3074` | `0.0048` |

## 평가

Fixed-control 값은 이전 A100 selected diagnostic band와 비슷하다. 하지만 operand를
다양하게 만든 두 variant는 훨씬 높게 측정되었다. Warp-static finite operand는
fixed 대비 `2.14x`, rotating operand set은 fixed 대비 `2.55x` 높다.

따라서 기존 fixed `1.0` operand 결과는 representative real FP16 matmul energy라기보다
fixed-pattern lower-bound diagnostic으로 해석하는 것이 적절하다.

## 산출물

| Artifact | 목적 |
|---|---|
| `runs.jsonl` | 30개 role run의 raw runner metadata |
| `summary.csv` | repeat별 baseline-subtracted 결과 |
| `condition_summary.csv` | operand condition별 mean/std/CI95 |
| `build_ptxas.log` | clean rebuild ptxas resource log |
| `resource_summary.txt` | target kernel register/spill 요약 |
| `static_sass_audit.csv` | MMA/LDG-like instruction static SASS count |
| `matrix_used.json` | 정확한 실험 matrix |
| `preflight_nvidia_smi.csv` | run 전 GPU telemetry metadata |

현재 Python 환경에 `matplotlib`이 없어 figure 생성은 건너뛰었다. CSV summary는
정상 생성되었다.

## Boundary

이 결과는 real-GEMM energy가 아니다. Register-resident HMMA diagnostic이며, NCU
counter validation이 없고 timed A/B global-memory load도 없다.
