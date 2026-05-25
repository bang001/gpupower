# Quick Verification Result

작성일: 2026-05-25

## 목적

시간을 오래 쓰지 않고, read/write energy decomposition이 의도한 구조로 실제 실행되는지 확인했다. 이 결과는 정밀 pJ/bit 산출용이 아니라 구현 및 실행 경로 검증용이다.

## 실행 조건

공통 조건:

- GPU: `NVIDIA H100 80GB HBM3`
- device: `0`
- targets: `50,100`
- repeats: `2`
- warmup repeats: `0`
- phase seconds: `1`
- idle seconds: `2`
- DRAM buffer: `4294967296` bytes
- L2 warm passes: `2`
- output dir: `reports/quick_verify_20260525_084052`

Read 조건:

- L2 cache op: `cg`
- DRAM cache op: `cs`

Write 조건:

- L2 cache op: `wb`
- DRAM cache op: `cs`
- write pattern: `address`

NCU 조건:

- read/write 각각 `dram` stage만 짧게 시도
- DRAM buffer: `1073741824` bytes
- launch count: `1`

## 전체 판정

| 항목 | 판정 | 의미 |
|---|---:|---|
| Read 4-stage energy run | PASS | `control_l2`, `l2`, `control_dram`, `dram` 모두 실행됨 |
| Write 4-stage energy run | PASS | `control_l2`, `l2`, `control_dram`, `dram` 모두 실행됨 |
| Read cache-op 전달 | PASS | `l2_read_cg`, `dram_read_cs`로 실행됨 |
| Write cache-op 전달 | PASS | `l2_write_wb_address`, `dram_write_cs_address`로 실행됨 |
| Read/write 산출물 생성 | PASS | summary, aggregate, fits, decomposition, quality, trace, metadata, png 생성됨 |
| NCU wrapper 실행 | PARTIAL | NCU는 프로세스에 붙었지만 counter 권한에서 실패함 |
| NCU counter 수집 | FAIL | `ERR_NVGPUCTRPERM` 때문에 kernel profiling 불가 |

## Read 결과

산출물 prefix:

- `read_energy_decomp_nvidia_h100_80gb_hbm3_20260525_084126_quick_verify_read_*`

Decomposition:

| component | pJ/bit |
|---|---:|
| `control_l2_loop` | 0.649 |
| `l2_read_total` | 3.729 |
| `control_dram_loop` | 1.129 |
| `dram_read_total` | 10.188 |
| `l2_over_control` | 3.080 |
| `dram_over_control` | 9.059 |
| `dram_global_over_l2` | 5.979 |

품질:

- OK: complete decomposition coverage.
- OK: 모든 phase sample 수 충분.
- OK: active phase P-state는 P0.
- OK: 온도 경고 없음.
- OK: DRAM buffer가 L2의 64배 이상.
- WARN: quick run이라 fit point가 2개뿐이다.
- WARN: quick run이라 repeat가 2개뿐이다.
- WARN: L2 warm pass가 2라 정밀 실험 권장값 4보다 작다.

판정: read는 요청한 구조대로 실행됐다. pJ/bit 값은 quick verification 값이므로 최종 수치로 쓰지 않는다.

## Write 결과

산출물 prefix:

- `write_energy_decomp_nvidia_h100_80gb_hbm3_20260525_084201_quick_verify_write_*`

Decomposition:

| component | pJ/bit |
|---|---:|
| `control_l2_loop` | 1.091 |
| `l2_write_total` | 8.079 |
| `control_dram_loop` | 1.898 |
| `dram_write_total` | 10.782 |
| `l2_over_control` | 6.989 |
| `dram_over_control` | 8.884 |
| `dram_global_over_l2` | 1.896 |

품질:

- OK: complete decomposition coverage.
- OK: 모든 phase sample 수 충분.
- OK: active phase P-state는 P0.
- OK: 온도 경고 없음.
- OK: DRAM buffer가 L2의 64배 이상.
- WARN: quick run이라 fit point가 2개뿐이다.
- WARN: quick run이라 repeat가 2개뿐이다.
- WARN: L2 warm pass가 2라 정밀 실험 권장값 4보다 작다.

판정: write는 요청한 구조대로 실행됐다. pJ/bit 값은 quick verification 값이므로 최종 수치로 쓰지 않는다.

## Read vs Write quick 비교

| metric | Read pJ/bit | Write pJ/bit | quick 비교 |
|---|---:|---:|---|
| L2 total | 3.729 | 8.079 | write가 더 크게 측정됨 |
| DRAM total | 10.188 | 10.782 | 비슷한 범위 |
| L2 over control | 3.080 | 6.989 | write store path가 더 크게 측정됨 |
| DRAM over control | 9.059 | 8.884 | 비슷한 범위 |
| DRAM global over L2 | 5.979 | 1.896 | quick 조건에서는 read 쪽 off-chip 추가분이 더 크게 나옴 |

주의: 이 비교는 기능 검증용이다. 최종 비교에는 기존 read full 조건과 같은 write full run이 필요하다.

## NCU 결과

Read NCU log:

- `quick_verify_read_ncu_dram.ncu.log`

Write NCU log:

- `quick_verify_write_ncu_dram.ncu.log`

결과:

- NCU CLI는 실행되고 대상 Python 프로세스에 attach됐다.
- 하지만 GPU performance counter 접근 권한이 없어 실패했다.
- 에러: `ERR_NVGPUCTRPERM`
- 따라서 `.ncu-rep` 파일은 생성되지 않았고, kernel counter는 수집되지 않았다.

판정: NCU wrapper 경로는 실행되지만 현재 시스템 권한으로는 NCU counter profiling이 완료되지 않는다. 관리자 권한 또는 `NVreg_RestrictProfilingToAdminUsers=0` 설정이 필요하다.

## 최종 결론

원한대로 read/write energy decomposition 실험이 실행되는지는 확인됐다. read와 write 모두 4-stage decomposition, cache-op 전달, 결과 파일 생성이 정상 동작했다. 다만 NCU counter profiling은 권한 문제로 완료되지 않았고, 이번 값은 짧은 검증 run이라 최종 pJ/bit 수치로 사용하면 안 된다.
