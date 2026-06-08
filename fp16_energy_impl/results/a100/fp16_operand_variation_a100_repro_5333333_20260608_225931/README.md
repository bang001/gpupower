# A100 FP16 Operand Variation 5.3M 재현 실험

날짜: 2026-06-08
GPU: NVIDIA A100-SXM4-80GB
GPU UUID: `GPU-dbb33614-0be2-bdd4-76e0-ee7700ee4386`
Launch shape: `threads=384`, `blocks/SM=4`
설정: `iters=5,333,333`, `unroll=8`, `suppress_output_store=true`
측정 상태: NCU 사용 불가, NVML-only diagnostic

## 목적

기존 `iters=5,333,333` operand variation 결과가 현재 GPU에서 재현되는지 확인했다.
같은 config인 `configs/fp16_operand_variation_a100.json`을 사용했고, matrix 전체를
5회 반복했다.

## 현재 GPU 재현 결과

| 조건 | Runs | Valid no-L2 | Elapsed mean | TFLOPS mean | pJ/FLOP mean | CI95 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed control | 5 | 5 | `5.909s` | `306.62` | `0.1211` | `0.0049` |
| Warp-static 4-set | 5 | 5 | `5.836s` | `310.46` | `0.2614` | `0.0062` |
| Warp-rotating 4-set | 5 | 5 | `6.197s` | `292.41` | `0.3167` | `0.0084` |

모든 조건에서 incremental energy는 양수였고, timed kernel metadata 기준 의도된
global memory access는 없었다. SM clock mean은 모든 조건에서 `1410 MHz`였다.

## 기존 5.3M 결과와 비교

| 조건 | 기존 5.3M pJ/FLOP | 현재 GPU 재현 pJ/FLOP | 변화율 |
|---|---:|---:|---:|
| Fixed control | `0.1204` | `0.1211` | `+0.60%` |
| Warp-static 4-set | `0.2575` | `0.2614` | `+1.53%` |
| Warp-rotating 4-set | `0.3074` | `0.3167` | `+3.02%` |

현재 GPU에서 같은 `iters=5,333,333` 조건을 다시 실행해도 기존 결과와 같은 범위로
재현된다. 특히 fixed control은 거의 동일하고, operand variation 조건도 기존보다
약간 높지만 큰 구조는 바뀌지 않는다.

## 현재 GPU 10M 결과와 비교

| 조건 | 현재 5.3M pJ/FLOP | 현재 10M pJ/FLOP | 변화율 |
|---|---:|---:|---:|
| Fixed control | `0.1211` | `0.1232` | `+1.71%` |
| Warp-static 4-set | `0.2614` | `0.2681` | `+2.55%` |
| Warp-rotating 4-set | `0.3167` | `0.3204` | `+1.18%` |

측정 window를 약 11초로 늘리면 `pJ/FLOP`이 약 `1-3%` 정도 높아졌지만, operand
variation이 fixed control보다 크게 높다는 결론은 그대로 유지된다.

## 해석

현재 GPU 기준으로도 fixed `1.0` operand 결과는 약 `0.12 pJ/FLOP`이고, warp별
finite operand set을 사용하면 약 `0.26 pJ/FLOP`, 시간에 따른 operand rotation까지
넣으면 약 `0.32 pJ/FLOP`로 올라간다.

따라서 기존 판단은 유지된다. Fixed-operand HMMA 값은 representative dynamic FP16
matmul energy라기보다 fixed-pattern lower-bound diagnostic으로 해석하는 것이
적절하다.
