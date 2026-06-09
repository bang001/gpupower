# FP16 Operand Variation 에너지 실험 계획

날짜: 2026-06-08
상태: diagnostic 실험 계획 및 1차 A100 실행 완료
Primary target: A100 `sm_80`
Profiler 상태: NCU hardware counter 사용 불가, 따라서 NVML-only diagnostic

1차 A100 결과는 [FP16_OPERAND_VARIATION_RESULTS.md](FP16_OPERAND_VARIATION_RESULTS.md)에
정리되어 있다. 실제 구현에서는 timed loop 안의 periodic rebase branch를 피하기
위해 작은 normal FP16 operand를 사용했고, 이 방식으로 accumulator가 finite 상태를
유지되도록 했다.

## 목표

현재 fixed-operand Tensor Core benchmark가 operand 값 변화가 거의 없어서
switching activity를 충분히 만들지 못하고, 그 결과 `pJ/FLOP`을 과소평가하는지
확인한다.

이 실험의 질문은 하나로 제한한다.

> A/B operand가 warp마다 다르고 시간에 따라 바뀌면, baseline-subtracted
> `pJ/FLOP`이 기존 fixed-register-operand HMMA loop 대비 의미 있게 변하는가?

목표는 broad sweep이나 final real-GEMM claim이 아니다. 기존 값이 constant-operand
lower bound인지 확인하기 위한 작은 diagnostic 실험이다.

## 현재 우려

현재 Tensor Core kernel은 real FP16 matmul보다 훨씬 제한된 형태다.

1. `tensor_mma_f16acc_kernel`은 A/B fragment를 kernel 내부 register constant로 둔다.
   기존 code에서 A/B half 값은 모두 `1.0`이다.
2. timed Tensor Core path는 global memory나 L2에서 큰 input matrix를 읽지 않는다.
   할당된 output buffer는 output/provenance 용도이며 A/B operand source가 아니다.
3. 긴 loop와 FP16 accumulation에서는 `1.0 x 1.0` 누적이 빠르게 FP16 saturation에
   도달할 수 있다. 이후 accumulator 값은 의도보다 훨씬 덜 dynamic해질 수 있다.
4. NCU가 없으므로 HMMA issue rate, L2/DRAM bytes, local memory spill,
   Tensor activity를 hardware counter로 직접 검증할 수 없다.

따라서 현재 값은 real application GEMM energy가 아니라 register-resident HMMA
diagnostic estimate로 남겨야 한다.

## 최소 실험 Matrix

우선 현재 A100 대표 launch shape 하나만 사용한다.

| 항목 | 설정 |
|---|---|
| GPU | A100-SXM4-80GB |
| Kernel family | `tensor_mma_f16acc` |
| Launch shape | `threads=384`, `blocks/SM=4` |
| Runtime window | measured row당 최소 5초 |
| Repeats | 초기 확인 3회, 최종 diagnostic 보고 5회 |
| Energy source | NVML total energy counter |
| Baseline | variant-matched no-HMMA structural baseline |

primary shape에서 의미 있는 operand effect가 보일 때만 `threads=192`, `blocks/SM=8`
shape를 추가 확인한다. 처음부터 full launch-shape sweep을 다시 시작하지 않는다.

## Operand 조건

실험 조건은 세 개만 둔다.

| 조건 | 설명 | 목적 |
|---|---|---|
| `fixed_ones_current` | 기존 동작. 모든 A/B fragment가 `1.0` constant 사용 | 기존 보고서와 맞는 control point |
| `warp_static_4set` | finite operand set 네 개를 만들고 global warp id로 선택. 한 warp는 timed loop 동안 같은 set 유지 | warp-to-warp data diversity가 energy에 영향을 주는지 확인 |
| `warp_rotating_4set_bounded` | 같은 네 개 set을 사용하되 warp id와 loop index에 따라 set을 rotate. Accumulator는 finite 상태를 유지 | 시간에 따른 operand switching activity 영향 확인 |

실제 1차 구현에서는 zero-only, denormal, NaN, Inf를 피하고 작은 normal FP16 값을
사용했다. timed loop 안에서 random 값을 생성하지 않고 deterministic packed half
constant를 사용한다. 이렇게 해야 실험이 operand data 변화에 집중되고 control
overhead가 과하게 섞이지 않는다.

### 실제 finite operand set

구현은 A/B fragment를 `uint32_t` packed half2 immediate로 전달한다. 아래 표에서
하나의 pair는 `[low16, high16]` 순서의 두 FP16 값이다. Fixed control은 모든
A/B operand가 `0x3c003c00`, 즉 half2 `(1.0, 1.0)`이다.

`warp_static_4set`에서는 `finite_operand_set4((tid >> 5) & 3)`을 사용한다.
따라서 global warp id modulo 4로 set이 선택되며, 한 warp 안에서는 timed loop
동안 같은 set을 유지한다. `threads=384` 조건에서는 block당 12 warps이므로
set 순서는 `0, 1, 2, 3, ...`로 반복된다.

| Set | A operands `a0..a3` | B operands `b0..b3` |
|---|---|---|
| 0 | `(+0.0078125, -0.0078125)`, `(+0.0078125, +0.00390625)`, `(+0.00390625, -0.0078125)`, `(-0.0078125, +0.0078125)` | `(+0.0078125, +0.00390625)`, `(+0.0078125, -0.0078125)`, `(+0.00390625, +0.0078125)`, `(-0.0078125, +0.00390625)` |
| 1 | `(+0.00390625, +0.005859375)`, `(+0.005859375, +0.00390625)`, `(+0.00390625, -0.00390625)`, `(-0.00390625, +0.005859375)` | `(-0.00390625, +0.00390625)`, `(+0.00390625, +0.005859375)`, `(+0.005859375, -0.00390625)`, `(+0.00390625, +0.00390625)` |
| 2 | `(-0.005859375, +0.0068359375)`, `(+0.0068359375, -0.0048828125)`, `(-0.005859375, +0.0078125)`, `(+0.0068359375, -0.005859375)` | `(+0.0078125, -0.0048828125)`, `(-0.0048828125, +0.0068359375)`, `(+0.0068359375, +0.0078125)`, `(-0.005859375, -0.0048828125)` |
| 3 | `(+0.0068359375, -0.00390625)`, `(-0.00390625, +0.0078125)`, `(+0.0068359375, -0.005859375)`, `(+0.0078125, +0.0068359375)` | `(-0.005859375, +0.0078125)`, `(-0.00390625, +0.0068359375)`, `(+0.0078125, -0.005859375)`, `(+0.0068359375, +0.0078125)` |

이 값들을 선택한 이유는 다음과 같다.

1. 모두 finite, non-zero, normal FP16 값이다. zero-only, subnormal, NaN, Inf
   동작을 피해서 Tensor Core datapath의 특수 케이스를 줄인다.
2. `1/256`, `1/128`, `3/512`, `7/1024`, `-5/1024`처럼 FP16에서 정확히
   표현되는 binary fraction을 사용했다. 따라서 host-side 초기화나 random
   generation 없이 deterministic register immediate로 재현할 수 있다.
3. 부호와 exponent/mantissa bit pattern을 set마다 섞어 all-ones pattern보다
   operand switching activity를 만들도록 했다.
4. 크기를 작게 잡아 `1.0 x 1.0` 반복 누적보다 FP16 accumulator saturation
   위험을 줄였다. 이 덕분에 timed loop 안에 periodic rebase branch를 넣지 않고도
   finite 상태를 유지하는 diagnostic을 구성할 수 있었다.

## Baseline 요구사항

각 operand 조건에는 반드시 matched baseline이 필요하다.

baseline은 같은 launch shape, loop count, set-selection logic, register pressure class,
output suppression behavior를 유지하되 HMMA instruction만 제거해야 한다. 특히
`warp_rotating_4set_bounded`는 indexing/register update 비용이 있으므로 기존
fixed-operand baseline과 비교하면 operand effect와 extra integer/control work가 섞인다.

따라서 보고서는 아래 pair를 비교해야 한다.

```text
fixed_ones_current          - fixed_ones_current_baseline
warp_static_4set            - warp_static_4set_baseline
warp_rotating_4set_bounded  - warp_rotating_4set_bounded_baseline
```

## NCU 없이 가능한 검증

NCU가 막혀 있으므로 검증 범위는 명확히 제한한다.

| Check | Requirement |
|---|---|
| Build/resource log | ptxas register 수를 기록하고 local-memory spill warning이 없는지 확인 |
| Static binary audit | `cuobjdump` 또는 `nvdisasm`가 있으면 timed test kernel에 HMMA instruction이 있고 의도된 global A/B load가 없는지 확인 |
| Runtime metadata | GPU UUID, clocks, power limit, driver, CUDA runtime, active process preflight 기록 |
| Energy sanity | test와 baseline 모두 NVML total energy counter 사용, incremental energy 양수 |
| Repeat stability | 최종 5회 반복에 대해 mean, std, CI95 보고 |
| Numeric sanity | accumulator bound strategy를 기록하고 Inf/NaN 상태를 피함 |

이 검증은 NCU를 대체하지 않는다. 결과가 안정적이어도 최종 표현은
`diagnostic NVML-only`로 유지한다.

## 판정 기준

크고 반복 가능한 차이만 해석한다.

| 관찰 | 해석 |
|---|---|
| `warp_static_4set`과 `warp_rotating_4set_bounded`가 `fixed_ones_current`의 약 10-15% 안에 머무름 | fixed operand가 현재 낮은 `pJ/FLOP`의 지배적 원인은 아닐 가능성 |
| `warp_static_4set`은 높지만 rotating은 static과 비슷함 | warp-to-warp diversity는 중요하지만 temporal switching은 주된 효과가 아닐 가능성 |
| `warp_rotating_4set_bounded`가 fixed와 static보다 명확히 높음 | 기존 fixed-operand 결과는 dynamic FP16 matmul representative가 아니라 lower-bound diagnostic일 가능성 |
| 결과가 baseline subtraction 후 0에 가깝거나 음수, 또는 variance가 큼 | 측정 또는 baseline이 해석 가능한 상태가 아님 |

단일 row의 높고 낮음만으로 보고서를 수정하지 않고, absolute change와 repeat spread를
함께 본다.

## 출력 산출물

결과는 아래와 같은 새 result directory에 저장한다.

```text
fp16_energy_impl/results/a100/fp16_operand_variation_a100_<date>/
```

필수 산출물:

| Artifact | 목적 |
|---|---|
| `summary.csv` | run별 test/baseline energy, elapsed time, FLOPs, `pJ/FLOP` |
| `condition_summary.csv` | operand condition별 mean/std/CI95 |
| `preflight.json` / `preflight.csv` | GPU/toolchain/process 상태 |
| `build_ptxas.log` | compile/resource evidence |
| `README.md` 또는 짧은 report | 결과 해석과 claim boundary |

`pJ/FLOP` 조건별 bar chart 하나면 충분하다. 이 실험에서는 큰 figure set을 만들지 않는다.

## Non-Goals

이 실험은 아래 항목을 포함하지 않는다.

1. Full thread/block sweep.
2. CUTLASS 또는 cuBLAS real GEMM 측정.
3. DRAM/L2 traffic energy 측정.
4. H100 또는 RTX 3090 cross-GPU 비교.
5. NCU counter validation.

operand variation이 `pJ/FLOP`을 의미 있게 증가시키면, 다음 단계는 별도의
real-GEMM calibration plan으로 분리한다. 이 문서를 그 큰 질문까지 확장하지 않는다.

## 권장 보고 문구

실험 후에는 아래처럼 표현한다.

```text
이 run은 register-resident HMMA loop에서 operand switching activity가 energy에 미치는
영향을 확인한 NVML-only diagnostic 실험이다. NCU counter가 없고 timed kernel이
global memory에서 A/B matrix를 읽지 않으므로 real GEMM energy를 증명하지 않는다.
결과는 이전 fixed-operand pJ/FLOP estimate를 lower-bound diagnostic으로 볼지 판단하는
근거로 사용한다.
```
