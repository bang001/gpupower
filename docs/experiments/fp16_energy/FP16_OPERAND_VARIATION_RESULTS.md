# FP16 Operand Variation 진단 실험 결과

날짜: 2026-06-08
GPU: NVIDIA A100-SXM4-80GB, GA100, `sm_80`
Launch shape: `threads=384`, `blocks/SM=4`
측정 상태: NCU 사용 불가, NVML-only diagnostic
결과 디렉터리: `../../../fp16_energy_impl/results/a100/fp16_operand_variation_a100_20260608_220327/`

## 질문

기존 fixed-register-operand HMMA loop의 `pJ/FLOP` 값이 A/B operand와
accumulator 값이 너무 정적으로 유지되어 실제보다 낮게 측정된 것인가?

## 구현

세 조건을 각각 5회 반복 측정했다.

| 조건 | Test kernel | Baseline kernel | Operand mode |
|---|---|---|---|
| Fixed control | `tensor_mma_f16acc` | `tensor_baseline_f16acc_fixed_ones` | 기존 fixed `1.0` A/B operand |
| Warp-static 4-set | `tensor_mma_f16acc_warp_static_4set` | `tensor_baseline_f16acc_warp_static_4set` | global warp id별로 네 개의 작은 normal FP16 operand set 중 하나 선택 |
| Warp-rotating 4-set | `tensor_mma_f16acc_warp_rotating_4set_bounded` | `tensor_baseline_f16acc_warp_rotating_4set_bounded` | 같은 네 개 set을 warp id와 outer loop index에 따라 rotate |

초기 계획에는 accumulator를 주기적으로 rebase하는 방법도 포함되어 있었다.
하지만 실제 구현에서는 timed loop 안에 branch가 들어가 해석을 흐리는 것을
피하기 위해 더 작은 normal FP16 operand를 사용했다. 이 방식은 선택한 5초
workload에서 accumulator가 finite 상태를 유지하도록 하면서, periodic rebase
overhead를 추가하지 않는다.

## 결과

| 조건 | Runs | Valid no-L2 | TFLOPS mean | pJ/FLOP mean | CI95 | Fixed 대비 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed control | 5 | 5 | `306.49` | `0.1204` | `0.0066` | `1.00x` |
| Warp-static 4-set | 5 | 5 | `310.49` | `0.2575` | `0.0029` | `2.14x` |
| Warp-rotating 4-set | 5 | 5 | `292.52` | `0.3074` | `0.0048` | `2.55x` |

모든 row는 test와 baseline 양쪽에서 NVML total energy counter를 사용했다.
모든 row에서 incremental energy는 양수였고, timed kernel metadata상 의도된
global memory access는 없었다. 측정 중 SM clock은 `1410 MHz`로 고정되었고
관측된 clock span은 `0 MHz`였다.

## Static Validation

`build_ptxas.log`와 `resource_summary.txt` 기준으로 측정 대상 unroll-8 kernel에는
local-memory spill이 없었다.

| Kernel class | Registers/thread | Spills |
|---|---:|---|
| Fixed HMMA test | 14 | 0 bytes |
| Operand-variation HMMA tests | 18-19 | 0 bytes |
| Matched operand baselines | 14 | 0 bytes |

`static_sass_audit.csv`에서는 HMMA test kernel에 MMA-like instruction이 있고,
matched baseline에는 MMA-like instruction이 없는 것으로 확인되었다. 또한 target
section 안에서 LDG-like instruction은 관측되지 않았다. 다만 SASS static audit은
binary 안의 `suppress_output_store=false` 경로까지 볼 수 있어 dormant final store
instruction을 셀 수 있다. 실제 측정 JSON metadata는 `suppress_output_store=true`와
timed-kernel global output store 없음으로 기록되어 있다.

## 해석

이번 결과는 기존 fixed-operand 값이 lower-bound diagnostic에 가깝다는 의심을
지지한다. fixed `1.0` operand 하나만 쓰는 조건에서는 `0.1204 pJ/FLOP`이
측정되었지만, warp별로 다른 finite FP16 operand set을 사용하자
`0.2575 pJ/FLOP`까지 증가했다. 같은 set을 시간에 따라 rotate하면
`0.3074 pJ/FLOP`까지 더 증가했다.

가장 중요한 관찰은 warp-static 4-set 조건이다. 이 조건은 fixed control과
비슷한 TFLOPS를 유지했는데도 `pJ/FLOP`이 크게 증가했다. 따라서 차이는 단순히
throughput 저하 때문만이 아니라 operand와 accumulator value pattern의 영향으로
보는 것이 타당하다. Rotating 조건은 static 조건보다 약 `19%` 더 높았으므로
temporal operand 변화도 영향을 준다. 하지만 가장 큰 변화는 all-ones/saturating
pattern에서 벗어나는 순간 발생했다.

## Claim Boundary

이 숫자는 여전히 real application GEMM energy가 아니다. timed kernel은
register-resident HMMA loop이며, A/B matrix를 global memory에서 읽지 않는다.
또한 NCU counter가 없으므로 HMMA issue rate, L2/DRAM bytes, local memory counter,
Tensor Core activity를 hardware counter로 검증하지 못했다.

따라서 기존 A100 fixed-operand 값은 representative dynamic FP16 matmul energy가
아니라 fixed-pattern lower-bound diagnostic으로 해석해야 한다.
