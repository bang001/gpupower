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

## 조건별 차이

| 비교 | 관측값 | 의미 |
|---|---:|---|
| Fixed control | `0.1204 pJ/FLOP` | 모든 warp가 같은 `1.0` A/B operand를 반복하므로 switching activity가 작을 수 있는 lower-bound control |
| Warp-static 4-set | `0.2575 pJ/FLOP`, fixed 대비 `2.14x` | warp별로 서로 다른 finite operand set을 쓰는 것만으로 energy가 크게 증가함. fixed all-ones pattern이 낮은 값을 만들었을 가능성을 직접 확인하는 핵심 비교 |
| Warp-rotating 4-set | `0.3074 pJ/FLOP`, fixed 대비 `2.55x`, static 대비 `1.19x` | 같은 operand set을 시간에 따라 rotate하면 static보다 더 높아짐. temporal operand switching도 영향을 주지만, 추가 operand 선택/register update 비용이 일부 섞일 수 있음 |

따라서 가장 중요한 비교는 fixed와 warp-static이다. Warp-static은 rotating보다
제어 변화가 적고 TFLOPS도 fixed와 비슷하게 유지되는데도 `pJ/FLOP`이 크게 높다.
이는 차이가 단순 throughput 저하나 실행 overhead 때문만이 아니라 operand value
pattern과 accumulator switching 차이에서 왔을 가능성을 강화한다.

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
