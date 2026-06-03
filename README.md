# GPU Power Experiments

This repository contains GPU power and energy experiments for AccelWattch-style modeling, with emphasis on FP16 Tensor Core energy estimation and DRAM/power-measurement methodology.

## Main Documentation

- [Documentation index](docs/README.md)
- [FP16 energy experiment index](docs/experiments/fp16_energy/README.md)
- [A100 FP16 energy report](docs/experiments/fp16_energy/A100_FP16_ENERGY_REPORT.md)
- [RTX 3090 FP16 results](docs/experiments/fp16_energy/RTX3090_FP16_RESULTS.md)
- [A100 vs RTX 3090 comparison](docs/experiments/fp16_energy/FP16_A100_RTX3090_COMPARISON.md)

## Current FP16 Result Boundary

The current A100 and RTX 3090 numbers are diagnostic `NVML total energy counter` estimates for logical `m16n16k16` FP16 Tensor Core input bits. They are not strict publishable hardware-counter claims because Nsight Compute performance counters are blocked by `ERR_NVGPUCTRPERM` in the available environments.

Raw experiment artifacts remain under [fp16_energy_impl/results](fp16_energy_impl/results/README.md).
