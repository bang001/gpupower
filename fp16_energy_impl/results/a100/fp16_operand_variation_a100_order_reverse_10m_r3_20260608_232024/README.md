# A100 FP16 Operand Variation 10M Reverse-Order 결과

날짜: 2026-06-08
GPU: NVIDIA A100-SXM4-80GB
GPU UUID: `GPU-dbb33614-0be2-bdd4-76e0-ee7700ee4386`
Launch shape: `threads=384`, `blocks/SM=4`
설정: `iters=10,000,000`, matrix repeat 3
조건 순서: `rotating -> static -> fixed`
측정 상태: NCU 사용 불가, NVML-only diagnostic

## 결과

| 조건 | Runs | Valid no-L2 | Elapsed mean | TFLOPS mean | pJ/FLOP mean | CI95 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed control | 3 | 3 | `11.080s` | `306.64` | `0.1176` | `0.0007` |
| Warp-static 4-set | 3 | 3 | `10.948s` | `310.31` | `0.2664` | `0.0001` |
| Warp-rotating 4-set | 3 | 3 | `11.626s` | `292.23` | `0.3299` | `0.0055` |

## Normal-Order 10M 대비

| 조건 | Normal pJ/FLOP | Reverse pJ/FLOP | 변화율 |
|---|---:|---:|---:|
| Fixed control | `0.1232` | `0.1176` | `-4.53%` |
| Warp-static 4-set | `0.2681` | `0.2664` | `-0.64%` |
| Warp-rotating 4-set | `0.3204` | `0.3299` | `+2.96%` |

## 해석

10M longer-window에서도 order-control 결론은 유지된다. Fixed control은 마지막에
실행해도 낮고, rotating 조건은 첫 번째로 실행해도 높다. 따라서 fixed/static/rotating
간 큰 차이는 measurement order bias가 아니라 operand pattern 차이로 보는 것이
타당하다.
