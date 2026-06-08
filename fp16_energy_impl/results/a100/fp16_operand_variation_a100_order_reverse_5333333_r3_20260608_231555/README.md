# A100 FP16 Operand Variation 5.3M Reverse-Order 결과

날짜: 2026-06-08
GPU: NVIDIA A100-SXM4-80GB
GPU UUID: `GPU-dbb33614-0be2-bdd4-76e0-ee7700ee4386`
Launch shape: `threads=384`, `blocks/SM=4`
설정: `iters=5,333,333`, matrix repeat 3
조건 순서: `rotating -> static -> fixed`
측정 상태: NCU 사용 불가, NVML-only diagnostic

## 결과

| 조건 | Runs | Valid no-L2 | Elapsed mean | TFLOPS mean | pJ/FLOP mean | CI95 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed control | 3 | 3 | `5.911s` | `306.53` | `0.1155` | `0.0060` |
| Warp-static 4-set | 3 | 3 | `5.836s` | `310.49` | `0.2534` | `0.0051` |
| Warp-rotating 4-set | 3 | 3 | `6.199s` | `292.30` | `0.3202` | `0.0063` |

## Normal-Order 5.3M 대비

| 조건 | Normal pJ/FLOP | Reverse pJ/FLOP | 변화율 |
|---|---:|---:|---:|
| Fixed control | `0.1211` | `0.1155` | `-4.62%` |
| Warp-static 4-set | `0.2614` | `0.2534` | `-3.09%` |
| Warp-rotating 4-set | `0.3167` | `0.3202` | `+1.10%` |

## 해석

Fixed를 마지막에 실행해도 낮은 값이 유지되고, rotating을 첫 번째로 실행해도 높은
값이 유지된다. 따라서 operand variation으로 인한 큰 `pJ/FLOP` 차이는 실행 순서나
thermal ordering만으로 설명되지 않는다.
