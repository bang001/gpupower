# A100 FP16 Operand Variation 10M Iteration 결과

날짜: 2026-06-08
GPU: NVIDIA A100-SXM4-80GB
Launch shape: `threads=384`, `blocks/SM=4`
설정 변경: `iters=5,333,333`에서 `iters=10,000,000`으로 증가
측정 상태: NCU 사용 불가, NVML-only diagnostic

## 질문

기존 약 5-6초 측정 window를 약 11초로 늘렸을 때 operand variation 결과와
`pJ/FLOP` 결론이 달라지는지 확인했다.

## 10M 결과

| 조건 | Runs | Valid no-L2 | Elapsed mean | TFLOPS mean | pJ/FLOP mean | CI95 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed control | 5 | 5 | `11.080s` | `306.61` | `0.1232` | `0.0054` |
| Warp-static 4-set | 5 | 5 | `10.952s` | `310.21` | `0.2681` | `0.0016` |
| Warp-rotating 4-set | 5 | 5 | `11.622s` | `292.32` | `0.3204` | `0.0025` |

모든 조건에서 incremental energy는 양수였고, timed kernel metadata 기준 의도된
global memory access는 없었다. SM clock mean은 모든 조건에서 `1410 MHz`,
clock span mean은 `0 MHz`였다.

## 기존 5.3M 결과와 비교

| 조건 | 5.3M pJ/FLOP | 10M pJ/FLOP | 변화율 | 5.3M elapsed | 10M elapsed |
|---|---:|---:|---:|---:|---:|
| Fixed control | `0.1204` | `0.1232` | `+2.32%` | `5.912s` | `11.080s` |
| Warp-static 4-set | `0.2575` | `0.2681` | `+4.11%` | `5.836s` | `10.952s` |
| Warp-rotating 4-set | `0.3074` | `0.3204` | `+4.23%` | `6.194s` | `11.622s` |

## 해석

10M iteration으로 window를 늘려도 큰 결론은 바뀌지 않았다. Fixed control은 여전히
약 `0.12 pJ/FLOP` 수준이고, operand variation 조건은 fixed 대비 크게 높다.
다만 window를 늘리면 세 조건 모두 `pJ/FLOP`이 약 `2-4%` 높아졌다. 이는 짧은
window의 우연한 샘플링 효과라기보다, 온도/전력 steady-state 또는 baseline
subtraction 조건 차이가 조금 반영된 것으로 볼 수 있다.

따라서 기존 결론은 유지한다. Fixed `1.0` operand 결과는 real dynamic FP16 matmul
energy라기보다 fixed-pattern lower-bound diagnostic으로 해석하는 것이 적절하다.
