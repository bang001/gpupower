# A100 FP16 pJ/bit 실험 검수 기록

작성일: 2026-06-03  
대상 결과: `results/fp16_matmul_pjbit_a100_20260603_031033`  
검수 범위: A100 FP16 Tensor Core `tensor_mma_f16acc` pJ/bit 실험 실행 정상성, 결과 품질, NCU 가능 여부

## 결론

A100에서 FP16 `tensor_mma_f16acc` vs `tensor_baseline_mov` 실험은 **NVML total energy counter 기반 diagnostic 실험으로는 정상 실행**되었다. 결과 파일, raw run 수, GPU metadata, schema, denominator, memory provenance, energy source, clock/temperature 상태가 모두 기대 조건과 일치한다.

단, 현재 vast.ai 컨테이너에서는 NCU performance counter 접근이 막혀 있어 **strict NCU-validated claim은 불가**하다. 따라서 보고값은 `NVML total energy counter 기반 diagnostic`으로 유지해야 하며, no-L2/no-DRAM/HMMA activity는 benchmark metadata와 software-side evidence까지만 확인된 상태다.

## 1. 실행 환경 검수

| 항목 | 결과 | 판정 |
|---|---|---|
| GPU | NVIDIA A100-SXM4-80GB | PASS |
| Compute capability | `8.0` | PASS |
| Architecture tag | Ampere / GA100 | PASS |
| Driver | `580.159.03` | PASS |
| CUDA runtime 표시 | `13.0` | PASS |
| Persistence mode | On | PASS |
| MIG | Disabled | PASS |
| 실험 후 active process | 없음 | PASS |
| 현재 idle temp/power | 30 C, 약 61 W | PASS |

검수 시점 `nvidia-smi`에서 GPU memory 사용량은 0 MiB이고 running process는 없었다. 실험 중 외부 compute process 간섭을 의심할 근거는 발견하지 못했다.

## 2. 빌드 검수

| 항목 | 결과 | 판정 |
|---|---|---|
| Build type | `Release` | PASS |
| CUDA compiler | `/usr/local/cuda/bin/nvcc` | PASS |
| CMake CUDA architecture | `80` | PASS |
| 실행 binary | `build/fp16_energy_bench` | PASS |

A100 대상인 `sm_80`으로 빌드되어 있으며, smoke test와 전체 matrix 실행에서 binary 실행 실패는 없었다.

## 3. Raw Run 완전성

기대 run 수는 다음과 같다.

```text
2 conditions * 2 roles(test/baseline) * 12 repeats = 48 raw runs
```

실제 `runs.jsonl`은 48개다.

| Condition | Role | Count | 판정 |
|---|---|---:|---|
| `matmul_tensor_mma_f16acc_vs_tensor_baseline` | baseline | 12 | PASS |
| `matmul_tensor_mma_f16acc_vs_tensor_baseline` | test | 12 | PASS |
| `matmul_tensor_mma_f32acc_vs_tensor_baseline` | baseline | 12 | PASS |
| `matmul_tensor_mma_f32acc_vs_tensor_baseline` | test | 12 | PASS |

모든 raw run에서 GPU metadata는 동일했다.

```text
device_name = NVIDIA A100-SXM4-80GB
compute_capability = 8.0
architecture_chip = ga100
sm_count = 108
blocks = 864
threads = 256
blocks_per_sm_requested = 8
unroll = 8
suppress_output_store = True
```

## 4. Schema / Denominator 검수

모든 raw run이 현재 schema를 사용했다.

| 항목 | Count | 판정 |
|---|---:|---|
| `schema_version = fp16-energy-bench-v2` | 48/48 | PASS |
| `nvml_timed_energy_counter` | 48/48 | PASS |
| `explicit_m16n16k16_denominator` | 48/48 | PASS |
| `strict_denominator_provenance` | 48/48 | PASS |
| `timed_kernel_memory_provenance` | 48/48 | PASS |

FP16 Tensor Core denominator는 benchmark JSON metadata에서 직접 기록되며, analyzer fallback이 아니다.

```text
mma_logical_shape = m16n16k16
mma_input_bits_per_logical_mma = 8192
mma_flops_per_logical_mma = 8192
matmul_denominator_source = bench_json_metadata
matmul_denominator_valid = True
```

## 5. Memory Provenance 검수

모든 raw run에서 timed kernel 내부의 의도된 global/L2 memory operation은 없음으로 기록되었다.

| Metadata | Count | 판정 |
|---|---:|---|
| `timed_kernel_global_input_loads = False` | 48/48 | PASS |
| `timed_kernel_global_output_stores = False` | 48/48 | PASS |
| `timed_kernel_has_intended_global_memory = False` | 48/48 | PASS |

주의: 이는 benchmark metadata와 implementation intent에 대한 검수다. 실제 hardware counter 기반 no-L2/no-DRAM 증명은 NCU가 필요하지만, 현재 platform 권한 제한 때문에 수행하지 못했다.

## 6. Energy Source 검수

모든 raw run에서 NVML total energy counter가 지원되고, energy delta가 양수였다.

| 항목 | Count | 판정 |
|---|---:|---|
| `nvml_energy_supported = True` | 48/48 | PASS |
| `nvml_energy_delta_j > 0` | 48/48 | PASS |
| test/baseline energy source | `nvml_total_energy_counter` | PASS |

A100 `tensor_mma_f16acc` 안정 구간의 counter/trace cross-check는 다음과 같다.

| 항목 | 평균 | 해석 |
|---|---:|---|
| test counter/trace ratio | `1.007` | 양호 |
| baseline counter/trace ratio | `0.945` | 허용 범위 내 |

`power.draw.average/instant`는 같은 이름의 API라도 architecture와 driver별 smoothing/window 의미가 다를 수 있으므로, trace는 primary source가 아니라 sanity check로만 사용했다.

## 7. FP16 f16acc 결과 품질

검수 대상인 `tensor_mma_f16acc` vs `tensor_baseline_mov`는 12/12개 row가 valid였다.

| 항목 | 결과 | 판정 |
|---|---:|---|
| summary rows | 12 | PASS |
| `valid_basic` | 12/12 | PASS |
| `valid_no_l2` | 12/12 | PASS(metadata 기준) |
| `pure_fp16_candidate_no_l2` | 12/12 | PASS(metadata 기준) |
| quality gate pass | 12/12 | PASS |
| clock span | `0 MHz` | PASS |
| max temp | 38-40 C | PASS |

수치 요약:

| Scope | pJ/bit | TFLOPS | Incremental energy | Incremental fraction |
|---|---:|---:|---:|---:|
| 전체 12회 | `0.1304 +/- 0.0235` | `307.22 +/- 0.05` | `59.09 +/- 10.66 J` | `0.229 +/- 0.040` |
| 안정 구간 마지막 10회 | `0.1469 +/- 0.0109` | `307.23 +/- 0.06` | `66.54 +/- 4.92 J` | `0.257 +/- 0.014` |

초기 2개 row는 pJ/bit가 `0.0461`, `0.0502`로 낮게 튀었다. test throughput과 clock은 안정적이므로 kernel 실행 실패는 아니고, baseline subtraction 및 cold-start/power state settling 편차로 해석하는 것이 타당하다. 대표값은 마지막 10회 안정 구간을 사용하는 것이 보수적이다.

## 8. f32acc 보조 조건 검수

`tensor_mma_f32acc` vs `tensor_baseline_f32`는 보조 diagnostic 조건이다. 이 조건은 12개 중 9개만 valid였고, 3개는 nonpositive incremental energy로 invalid였다.

| 항목 | 결과 | 판정 |
|---|---:|---|
| summary rows | 12 | 참고 |
| `valid_basic` | 9/12 | 경고 |
| `invalid_or_nonpositive_increment` | 3/12 | 경고 |
| mean pJ/bit 전체 | `0.0112 +/- 0.0160` | 해석 부적합 |

따라서 f32acc 결과는 이번 요청의 FP16 f16acc pJ/bit 결론에 사용하지 않는 것이 맞다.

## 9. Quality Gate 해석

`quality_gates.csv`는 총 24 row이며, quality pass는 21개다.

| 구분 | Rows | Quality pass | 해석 |
|---|---:|---:|---|
| f16acc pair | 12 | 12 | 사용 가능 diagnostic |
| f32acc pair | 12 | 9 | 보조 diagnostic, 일부 invalid |

`target_pass`는 0개다. 이는 실험이 thread sweep 형태가 아니라 fixed matrix 형태이기 때문이다. `target_pass` 부재는 A100 f16acc pair 실행 실패를 의미하지 않는다. 단, 최종 architecture comparison의 selected strict target으로 자동 채택하려면 thread sweep 또는 별도 target selection metadata가 필요하다.

Quality gate warning으로 baseline elapsed time이 1초 warning threshold보다 짧다는 메시지가 있다. f16acc baseline elapsed는 약 0.428초로 minimum threshold 0.25초는 넘기므로 fail은 아니다. 다만 더 낮은 noise를 원하면 baseline `repeats` 또는 `iters`를 늘리는 것이 좋다.

## 10. NCU / Vast.ai 제한 검수

현재 플랫폼은 vast.ai 컨테이너로 보이며, NCU performance counter profiling은 불가능한 상태다.

확인 사항:

```text
uid=0(root)
```

컨테이너 내부 사용자는 root지만 effective capability에는 `CAP_SYS_ADMIN`이 없다.

```text
0x00000000a80425fb = cap_chown, cap_dac_override, cap_fowner, cap_fsetid,
                     cap_kill, cap_setgid, cap_setuid, cap_setpcap,
                     cap_net_bind_service, cap_net_raw, cap_sys_chroot,
                     cap_mknod, cap_audit_write, cap_setfcap
```

NVIDIA driver는 profiling counter를 admin-only로 제한한다.

```text
RmProfilingAdminOnly: 1
RegistryDwords: ""
RegistryDwordsPerDevice: ""
```

NCU는 process attach까지는 성공하지만 counter 접근에서 실패한다.

```text
==PROF== Connected to process .../build/fp16_energy_bench
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
==PROF== Disconnected
```

따라서 현재 환경에서는 다음 strict evidence를 만들 수 없다.

1. HMMA Tensor Core activity counter evidence.
2. L2 bytes read/write counter evidence.
3. DRAM bytes read/write counter evidence.
4. local memory spill counter evidence.

이는 코드/실험 실패가 아니라 host/container 권한 제한이다. 해결하려면 vast.ai 인스턴스를 `CAP_SYS_ADMIN` 또는 privileged/profiling-enabled 설정으로 재시작하거나, host NVIDIA module setting에서 profiling restriction을 해제해야 한다.

## 11. 최종 판정

| 검수 항목 | 판정 |
|---|---|
| A100에서 benchmark 실행 | PASS |
| A100 `sm_80` 빌드 | PASS |
| raw run 완전성 | PASS |
| f16acc schema/denominator | PASS |
| f16acc energy source | PASS |
| f16acc clock/thermal 안정성 | PASS |
| f16acc quality gate | PASS, 12/12 |
| f32acc 보조 조건 | PARTIAL, 9/12 valid |
| NCU strict validation | BLOCKED by platform permission |
| 최종 strict claim 가능 여부 | NO |
| diagnostic claim 가능 여부 | YES |

따라서 현재 보고서의 A100 대표값 `0.1469 +/- 0.0109 pJ/bit`은 타당한 diagnostic estimate로 유지할 수 있다. 다만 최종 논문/공식 비교 표에는 `NCU validation unavailable on vast.ai due to ERR_NVGPUCTRPERM` 조건을 명시해야 한다.
