# Existing Log Verification

작성일: 2026-05-25

## 목적

기존 로그만 사용해서 read/write energy decomposition이 사용자가 원한 방향대로 구현 및 실행됐는지 짧게 검증한다.

## 검증 계획

1. `reports`의 최신 read full run 산출물을 확인한다.
2. `/tmp/write_decomp_*`의 write 로그가 full 실험인지 smoke 실험인지 구분한다.
3. read/write 코드에서 cache-op, stage 구성, NCU wrapper가 실제로 구현됐는지 확인한다.
4. `.ncu-rep` / `.ncu.log` 산출물이 남아 있는지 확인한다.
5. read/write 비교가 가능한 상태인지 판정한다.

## 결론

| 항목 | 판정 | 근거 |
|---|---:|---|
| Read full 실험 | PASS | 최신 run에 4개 stage와 decomposition 결과가 모두 존재함 |
| Read cache 고려 | PASS | read `--l2-cache-op/--dram-cache-op`가 `ca,cg,cs`를 지원함 |
| Write 구현 | PASS | write 4-stage, store cache-op, write-pattern, wrapper가 구현됨 |
| Write full 실험 | NOT DONE | 기존 write 로그는 `/tmp` smoke run이며 full 비교용 조건이 아님 |
| NCU 프로파일링 결과 | NOT DONE | `reports`와 `/tmp`에서 `.ncu-rep`, `.ncu.log` 없음 |
| Read vs Write 정량 비교 | WAIT | write full run과 NCU 검증 로그가 있어야 비교 가능 |

## Read 로그 검증

대상:

- `reports/nvidia_h100_80gb_hbm3_202605250804/read_energy_decomp_nvidia_h100_80gb_hbm3_20260525_080440_h100_*`

실행 조건:

- device: `0`
- GPU: `NVIDIA H100 80GB HBM3`
- targets: `50,75,100`
- repeats: `5`
- warmup repeats: `1`
- phase seconds: `15`
- idle seconds: `15`
- L2 cache op: `cg`
- DRAM cache op: `cs`
- DRAM buffer: `8589934592` bytes
- blocks: `4224`, threads: `256`, SM count: `132`

로그 구조:

- `summary.csv`: 61 lines = header 1 + 60 phases
- 60 phases = 4 stages x 3 targets x 5 repeats
- `fits.csv`: `control_l2`, `l2`, `control_dram`, `dram` 모두 존재
- `decomposition.csv`: read decomposition component 모두 존재

Read decomposition 결과:

| component | pJ/bit | 의미 |
|---|---:|---|
| `control_l2_loop` | 1.678 | L2-sized control loop 기본 비용 |
| `l2_read_total` | 4.225 | L2-resident read 전체 보드 레벨 비용 |
| `control_dram_loop` | 1.101 | DRAM-sized control loop 기본 비용 |
| `dram_read_total` | 11.636 | DRAM streaming read 전체 보드 레벨 비용 |
| `l2_over_control` | 2.546 | control을 뺀 L2 read 경로 증가분 |
| `dram_over_control` | 10.535 | control을 뺀 DRAM global read 경로 증가분 |
| `dram_global_over_l2` | 7.989 | L2 경로를 제외한 off-chip/DRAM 추가분 |

품질 판정:

- PASS: 4개 stage fit이 모두 있음.
- PASS: 각 stage fit의 R2가 `0.999877` 이상임.
- PASS: active phase가 P0, 온도 경고 없음.
- WARN: warmup 제외 fit repeat가 4개라 median/IQR 안정성은 약함.
- WARN: clock/power limit setup을 강제하지 않았음.
- WARN: idle low-clock sample이 없어 idle baseline은 all idle sample로 fallback됨.

요약: read 실험은 사용자가 준 조건대로 정상 완료된 것으로 봐도 된다. 다만 논문/최종 표에 쓰려면 repeat를 늘리고 clock 고정 조건을 맞추는 것이 좋다.

## Write 로그 검증

대상:

- `/tmp/write_decomp_full_smoke/write_energy_decomp_nvidia_h100_80gb_hbm3_20260525_074204_full_smoke_*`

실행 조건:

- targets: `50,100`
- repeats: `1`
- phase seconds: `0.2`
- idle seconds: `0.1`
- DRAM buffer: `67108864` bytes
- L2 buffer: `4194304` bytes
- L2 cache op: `wb`
- DRAM cache op: `cs`
- write pattern: `address`

로그 구조:

- `summary.csv`: 9 lines = header 1 + 8 phases
- 8 phases = 4 stages x 2 targets x 1 repeat
- `fits.csv`: 4개 stage 모두 존재하지만 2-point fit임
- `quality.csv`: repeat 부족, idle IQR 큼, DRAM buffer < 64x L2 경고

Write smoke 결과:

| component | pJ/bit | 사용 가능 여부 |
|---|---:|---|
| `l2_write_total` | 5.250 | smoke 값, 비교용으로 사용 금지 |
| `dram_write_total` | 8.818 | smoke 값, 비교용으로 사용 금지 |
| `l2_over_control` | 3.209 | smoke 값, 비교용으로 사용 금지 |
| `dram_over_control` | 7.250 | smoke 값, 비교용으로 사용 금지 |
| `dram_global_over_l2` | 4.041 | smoke 값, 비교용으로 사용 금지 |

요약: write 코드는 실행되고 decomposition 파일도 생성된다. 하지만 이 로그는 짧은 smoke run이라 read full run과 정량 비교하면 안 된다.

## 코드 구현 확인

Read:

- `dram_energy_decomp_cupy.py`
  - `decomp_stream_read_ca` 구현됨.
  - `--l2-cache-op`, `--dram-cache-op` choices가 `ca,cg,cs`.
  - `--only-stage`는 NCU 검증용 단일 stage 실행으로 동작.
- `run_energy_decomp_ncu.sh`
  - read stage별 NCU 실행 wrapper가 있음.
  - cache op 인자를 app command로 전달함.

Write:

- `dram_write_energy_decomp_cupy.py`
  - `decomp_stream_write_wb/cg/cs/wt` 구현됨.
  - `--l2-cache-op`, `--dram-cache-op` choices가 `wb,cg,cs,wt`.
  - `--write-pattern` choices가 `zero,const,address,random,toggle`.
  - `make_specs()`에서 `control_l2`, `l2`, `control_dram`, `dram` 4-stage를 구성함.
  - `l2_over_control`, `dram_over_control`, `dram_global_over_l2` 계산 경로가 있음.
- `run_write_energy_decomp.sh`
  - full write launcher가 있음.
- `run_write_energy_decomp_ncu.sh`
  - write stage별 NCU 실행 wrapper가 있음.
  - `--write-pattern`, `--l2-cache-op`, `--dram-cache-op`를 전달함.

요약: 사용자가 요청한 "read와 write 모두 cache를 고려한 구현"은 코드 수준에서 반영되어 있다.

## NCU 확인

검색 범위:

- `reports`
- `/tmp`

검색 결과:

- `.ncu-rep`: 없음
- `.ncu.log`: 없음

판정:

- NCU wrapper는 구현되어 있지만, 현재 남아 있는 기존 로그만 보면 NCU profiling은 완료됐다고 볼 수 없다.
- 따라서 L2 hit/miss, DRAM physical bytes, store writeback 여부는 아직 NCU counter로 검증되지 않았다.

## Read vs Write 비교 가능 여부

현재 가능한 비교:

- 코드 구조 비교: 가능
- smoke 실행 여부 비교: 가능
- read full 결과 설명: 가능

현재 불가능한 비교:

- read full vs write full pJ/bit 정량 비교
- read/write cache-op별 물리 byte 비교
- NCU 기반 L2/DRAM residency 검증

필요한 다음 실행:

```bash
./run_write_energy_decomp.sh \
  --device 0 \
  --tag h100_write \
  --dram-buf-bytes 8589934592 \
  --targets 50 75 100 \
  --repeats 5 \
  --warmup-repeats 1 \
  --phase-seconds 15 \
  --idle-seconds 15 \
  --idle-settle-seconds 5 \
  --window-ms 1000 \
  --gap-seconds 1 \
  --l2-warm-passes 4 \
  --fit-aggregate median \
  --dram-cache-op cs \
  --l2-cache-op wb \
  --write-pattern address \
  --out-dir reports
```

그 다음 NCU 검증:

```bash
./run_energy_decomp_ncu.sh --device 0 --tag h100_read_ncu --out-dir reports
./run_write_energy_decomp_ncu.sh --device 0 --tag h100_write_ncu --out-dir reports
```

## 최종 판정

Read full 실험은 제대로 완료됐다. Write는 구현과 smoke 실행까지는 확인됐지만, full 조건 실험과 NCU 프로파일링 로그가 아직 없다. 따라서 "원한대로 구현되었는가"에 대한 답은 코드 기준 PASS, 기존 로그 기준 read PASS / write full NOT DONE / NCU NOT DONE이다.
