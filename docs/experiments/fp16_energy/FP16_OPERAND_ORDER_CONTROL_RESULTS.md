# FP16 Operand Variation Order-Control 결과

날짜: 2026-06-08
GPU: NVIDIA A100-SXM4-80GB
GPU UUID: `GPU-dbb33614-0be2-bdd4-76e0-ee7700ee4386`
Launch shape: `threads=384`, `blocks/SM=4`
측정 상태: NCU 사용 불가, NVML-only diagnostic

## 목적

기존 operand variation 실험은 항상 `fixed -> static -> rotating` 순서로 진행되었다.
이 때문에 thermal steady-state 또는 실행 순서가 fixed/static/rotating 차이를 만든
것인지 확인할 필요가 있었다.

이번 order-control은 조건 순서를 반대로 바꿨다.

```text
rotating -> static -> fixed
```

5.3M iteration과 10M iteration 양쪽을 모두 실행했다. Reverse-order run은 빠른
확인을 위해 matrix repeat를 3회로 제한했다. 비교 기준 normal-order run은 같은 GPU에서
이미 수행한 repeat 5 결과다.

## 결과 요약

| 조건 | Window | Normal order pJ/FLOP | Reverse order pJ/FLOP | 변화율 |
|---|---|---:|---:|---:|
| Fixed control | 5.3M | `0.1211` | `0.1155` | `-4.62%` |
| Warp-static 4-set | 5.3M | `0.2614` | `0.2534` | `-3.09%` |
| Warp-rotating 4-set | 5.3M | `0.3167` | `0.3202` | `+1.10%` |
| Fixed control | 10M | `0.1232` | `0.1176` | `-4.53%` |
| Warp-static 4-set | 10M | `0.2681` | `0.2664` | `-0.64%` |
| Warp-rotating 4-set | 10M | `0.3204` | `0.3299` | `+2.96%` |

## Reverse-Order Raw Summary

| 조건 | Window | Runs | Valid no-L2 | Elapsed mean | TFLOPS mean | pJ/FLOP mean | CI95 |
|---|---|---:|---:|---:|---:|---:|---:|
| Fixed control | 5.3M | 3 | 3 | `5.911s` | `306.53` | `0.1155` | `0.0060` |
| Warp-static 4-set | 5.3M | 3 | 3 | `5.836s` | `310.49` | `0.2534` | `0.0051` |
| Warp-rotating 4-set | 5.3M | 3 | 3 | `6.199s` | `292.30` | `0.3202` | `0.0063` |
| Fixed control | 10M | 3 | 3 | `11.080s` | `306.64` | `0.1176` | `0.0007` |
| Warp-static 4-set | 10M | 3 | 3 | `10.948s` | `310.31` | `0.2664` | `0.0001` |
| Warp-rotating 4-set | 10M | 3 | 3 | `11.626s` | `292.23` | `0.3299` | `0.0055` |

모든 reverse-order row는 NVML total energy counter를 사용했고, incremental energy가
양수였으며, analyzer 기준 `valid no-L2`로 분류되었다. SM clock mean은 모든 조건에서
`1410 MHz`였고, timed-kernel metadata상 의도된 global memory access는 없었다.

## Ratio 비교

| Run | Static / Fixed | Rotating / Fixed | Rotating / Static |
|---|---:|---:|---:|
| Normal 5.3M | `2.16x` | `2.62x` | `1.21x` |
| Reverse 5.3M | `2.19x` | `2.77x` | `1.26x` |
| Normal 10M | `2.18x` | `2.60x` | `1.20x` |
| Reverse 10M | `2.27x` | `2.81x` | `1.24x` |

## 해석

Order-control 결과는 thermal/order bias가 fixed/static/rotating 간 큰 차이를 만든다는
설명을 지지하지 않는다.

가장 중요한 관찰은 fixed control을 마지막에 실행해도 여전히 낮고, rotating 조건을
첫 번째로 실행해도 여전히 높다는 점이다. 실제로 reverse-order에서 fixed는 normal보다
오히려 약 `4.5%` 낮았고, rotating은 normal보다 약 `1-3%` 높았다. 따라서 기존 결과의
큰 차이, 즉 fixed `~0.12 pJ/FLOP` 대비 static `~0.26 pJ/FLOP`, rotating `~0.32 pJ/FLOP`
구조는 실행 순서만으로 설명되지 않는다.

다만 reverse-order run은 repeat 3이므로 CI 해석은 repeat 5 normal-order보다 보수적으로
봐야 한다. 이 실험은 order bias가 주된 원인인지 확인하는 diagnostic control이며,
real GEMM energy claim을 위한 실험은 아니다.

## 결론

현재까지의 A100 결과는 다음 해석을 강화한다.

```text
fixed 1.0 register-resident HMMA 값은 representative dynamic FP16 matmul energy가 아니라
fixed-pattern lower-bound diagnostic으로 보는 것이 적절하다.
```

Operand variation을 넣었을 때의 energy 증가는 같은 GPU 재현, 10M longer-window,
reverse-order control 모두에서 유지되었다.
