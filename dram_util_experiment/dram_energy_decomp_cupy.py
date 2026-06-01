#!/usr/bin/env python3
"""Read-only GPU memory-path energy decomposition via CuPy + NVML.

v6 additions: visualization-oriented plots for fit evidence, method logic,
and thermal/clock diagnostics.

v5 additions: partial --only-stage decomposition guard, longer idle settle,
stronger L2 warm default, and temperature quality warnings.

v4 additions: defensive only-stage handling, idle hysteresis filtering,
fit quality guards, cache-op-aware notes, and NaN-safe plotting/metadata.

v3 additions: optional nvidia-smi clock/persistence setup, robust repeat/median
fitting, streaming cache-op selection, an optional compute-only calibration stage,
and explicit Nsight Compute-friendly single-stage execution.

This is a second-generation experiment for the case where a plain DRAM
streaming pJ/bit number is too coarse.  It keeps the useful parts of
`dram_pjbit_cupy.py` (NVRTC preload, NVML polling, plotting helpers), but
changes the experiment design:

  * Use a read-only workload so writes, compression, dirty L2 writeback, and
    store-allocation effects do not contaminate the first pass.
  * Measure four matched stages at multiple active bandwidth points:
        control_l2   : legacy name for the L2 no-read loop baseline
        l2           : same global read instruction over an L2-resident buffer
        control_dram : legacy name for the DRAM no-read loop baseline
        dram         : same global read instruction over a buffer >> L2
  * Fit average board/GPU power versus measured nominal bandwidth using only
    active points.  This avoids treating a P8/P12 idle baseline as the active
    P0 baseline.  v3 can aggregate repeated phase measurements by median before
    fitting so one noisy phase does not dominate the slope.
  * Report lower-bound separations:
        L2 read increment             ~= SM/RF/LSU/L2 read-path increment
        DRAM read increment total     ~= whole DRAM-stream read path above matched baseline
        Post-L2 off-chip increment    ~= off-chip miss increment: MC + GPU PHY + HBM PHY/core
        compute-only          ~= optional no-global-memory SM/FMA dynamic reference
        vendor-HBM subtract ~= optional external HBM stack prior, not measured

Important limitation: NVIDIA NVML does not expose a DRAM-rail-only sensor on
ordinary datacenter GPUs.  Pure HBM stack energy must come from external rail
instrumentation, vendor pJ/bit assumptions, or an NDA-level model.  The optional
`--hbm-pjbit` argument only subtracts such an external prior from the board-level
slope; it does not magically measure HBM-only power.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
try:
    import nvtx
except ImportError:
    class DummyNvtx:
        @staticmethod
        def annotate(*args, **kwargs):
            import contextlib
            @contextlib.contextmanager
            def dummy_context():
                yield
            return dummy_context()
    nvtx = DummyNvtx()


import dram_pjbit_cupy as base

cp = base.cp
plt = base.plt
pynvml = base.pynvml


KERNEL_CODE = r"""
extern "C" __global__
void decomp_control_read(uint4* __restrict__ sink,
                         unsigned long long n,
                         int passes) {
    unsigned long long tid    = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0U, 0U, 0U, 0U);

    for (int p = 0; p < passes; ++p) {
        for (unsigned long long i = tid; i < n; i += stride) {
            unsigned int lo = (unsigned int)i;
            unsigned int hi = (unsigned int)(i >> 32);
            // Keep enough integer/control work that compiler cannot collapse the loop,
            // while avoiding any intentional global memory traffic in the hot loop.
            acc.x ^= lo + (unsigned int)p;
            acc.y += hi ^ 0x9e3779b9U;
            acc.z ^= (lo * 0x85ebca6bU) + (unsigned int)p;
            acc.w += (hi * 0xc2b2ae35U) ^ lo;
        }
    }

    // One sink element per warp avoids the heavy many-thread-to-1024-location
    // write race in the older hierarchy script.  This tiny write is outside the
    // intended denominator and is identical across stages.
    unsigned int warp_in_block = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        sink[(unsigned long long)blockIdx.x * (blockDim.x >> 5) + warp_in_block] = acc;
    }
}


extern "C" __global__
void decomp_stream_read_ca(const uint4* __restrict__ in,
                        uint4* __restrict__ sink,
                        unsigned long long n,
                        int passes) {
    unsigned long long tid    = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0U, 0U, 0U, 0U);

    for (int p = 0; p < passes; ++p) {
        for (unsigned long long i = tid; i < n; i += stride) {
            uint4 v;
            const unsigned int* ptr = reinterpret_cast<const unsigned int*>(in + i);
            // ca: cache at all available cache levels.  On NVIDIA GPUs this can
            // involve L1/TEX as well as L2, so use NCU counters to confirm where
            // hits actually land for a given working set.
            asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                         : "l"(ptr));
            acc.x ^= v.x;
            acc.y += v.y;
            acc.z ^= v.z + (unsigned int)p;
            acc.w += v.w ^ (unsigned int)i;
        }
    }

    unsigned int warp_in_block = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        sink[(unsigned long long)blockIdx.x * (blockDim.x >> 5) + warp_in_block] = acc;
    }
}

extern "C" __global__
void decomp_stream_read_cg(const uint4* __restrict__ in,
                        uint4* __restrict__ sink,
                        unsigned long long n,
                        int passes) {
    unsigned long long tid    = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0U, 0U, 0U, 0U);

    for (int p = 0; p < passes; ++p) {
        for (unsigned long long i = tid; i < n; i += stride) {
            uint4 v;
            const unsigned int* ptr = reinterpret_cast<const unsigned int*>(in + i);
            // cg: bypass L1 and allocate/probe through L2.  L2 still cannot be
            // disabled from normal CUDA; the DRAM phase relies on reuse distance >> L2.
            asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                         : "l"(ptr));
            acc.x ^= v.x;
            acc.y += v.y;
            acc.z ^= v.z + (unsigned int)p;
            acc.w += v.w ^ (unsigned int)i;
        }
    }

    unsigned int warp_in_block = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        sink[(unsigned long long)blockIdx.x * (blockDim.x >> 5) + warp_in_block] = acc;
    }
}

extern "C" __global__
void decomp_stream_read_cs(const uint4* __restrict__ in,
                           uint4* __restrict__ sink,
                           unsigned long long n,
                           int passes) {
    unsigned long long tid    = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0U, 0U, 0U, 0U);

    for (int p = 0; p < passes; ++p) {
        for (unsigned long long i = tid; i < n; i += stride) {
            uint4 v;
            const unsigned int* ptr = reinterpret_cast<const unsigned int*>(in + i);
            // cs: streaming/evict-first hint.  This is NOT a guaranteed L2 bypass,
            // but it reduces cache residency pressure for the DRAM-stream phase.
            asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                         : "l"(ptr));
            acc.x ^= v.x;
            acc.y += v.y;
            acc.z ^= v.z + (unsigned int)p;
            acc.w += v.w ^ (unsigned int)i;
        }
    }

    unsigned int warp_in_block = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        sink[(unsigned long long)blockIdx.x * (blockDim.x >> 5) + warp_in_block] = acc;
    }
}

extern "C" __global__
void decomp_compute_fma(uint4* __restrict__ sink,
                        unsigned long long n,
                        int passes) {
    unsigned long long tid    = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    float x = 1.0f + (float)(threadIdx.x & 31) * 0.0001f;
    float y = 1.000001f;
    float z = 0.999999f;

    for (int p = 0; p < passes; ++p) {
        for (unsigned long long i = tid; i < n; i += stride) {
            // Intentional no-global-memory compute reference.  Four dependent FMAs
            // keep the loop from becoming a pure integer/control microbenchmark.
            asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(x) : "f"(x), "f"(y), "f"(z));
            asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y) : "f"(y), "f"(z), "f"(x));
            asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(z) : "f"(z), "f"(x), "f"(y));
            asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(x) : "f"(x), "f"(y), "f"(z));
        }
    }

    unsigned int warp_in_block = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        sink[(unsigned long long)blockIdx.x * (blockDim.x >> 5) + warp_in_block] =
            make_uint4(__float_as_uint(x), __float_as_uint(y),
                       __float_as_uint(z), (unsigned int)n);
    }
}

"""


@dataclass(frozen=True)
class WorkloadSpec:
    stage: str
    name: str
    kernel_name: str
    buffer_kind: str
    does_global_read: bool
    cache_op: str
    is_compute: bool
    n_u4: int
    buf_bytes: int


@dataclass
class PhaseRow:
    repeat: int
    stage: str
    target_pct: int
    phase: str
    t0_s: float
    t1_s: float
    wall_s: float
    launches: int
    passes_per_launch: int
    nominal_bytes: int
    nominal_bandwidth_gbs: float
    total_energy_j: float
    idle_energy_j: float
    dynamic_energy_j: float
    avg_power_w: float
    dynamic_power_w: float
    pj_per_nominal_bit: float
    samples: int
    power_std_w: float
    power_p05_w: float
    power_p50_w: float
    power_p95_w: float
    sm_clock_mhz_mean: float
    mem_clock_mhz_mean: float
    temp_gpu_c_mean: float
    pstate_p50: float
    gpu_util_pct_mean: float
    mem_util_pct_mean: float
    avg_power_instant_w: float
    avg_power_average_w: float
    power_instant_samples: int
    power_average_samples: int


@dataclass
class FitResult:
    stage: str
    n_points: int
    targets: str
    slope_w_per_gbs: float
    intercept_w: float
    r2: float
    max_abs_residual_w: float
    pj_per_nominal_bit: float


@dataclass
class DecompResult:
    component: str
    pj_per_bit: float
    note: str
    legacy_component: str = ""


REQUIRED_DECOMP_STAGES = ("control_l2", "l2", "control_dram", "dram")
STAGE_ORDER = ("control_l2", "l2", "control_dram", "dram", "compute")

STAGE_COLORS = {
    "control_l2": "#7f7f7f",
    "l2": "#4c78a8",
    "control_dram": "#bab0ac",
    "dram": "#e45756",
    "compute": "#54a24b",
}

STAGE_LABELS = {
    "control_l2": "L2 loop\nbaseline",
    "l2": "L2\nresident",
    "control_dram": "DRAM loop\nbaseline",
    "dram": "DRAM\nstream",
    "compute": "compute\nreference",
}

COMPONENT_LABELS = {
    "l2_loop_baseline": "L2 loop\nbaseline",
    "l2_read_total": "L2 read\ntotal",
    "dram_loop_baseline": "DRAM loop\nbaseline",
    "dram_read_total": "DRAM read\ntotal",
    "l2_read_increment": "L2 read\nincrement",
    "dram_read_increment_total": "DRAM read\nincrement total",
    "post_l2_offchip_increment": "Post-L2\noff-chip inc.",
    "compute_only_reference": "compute\nreference",
    "hbm_stack_external_prior": "external\nHBM prior",
    "gpu_side_after_l2_est": "GPU side\nafter L2",
    "gpu_side_whole_read_path_est": "GPU side\nwhole path",
    "decomposition_incomplete": "incomplete",
}


def stage_color(stage: str) -> str:
    return STAGE_COLORS.get(stage, "#8c8c8c")


def stage_label(stage: str) -> str:
    return STAGE_LABELS.get(stage, stage.replace("_", "\n"))


def component_label(component: str) -> str:
    return COMPONENT_LABELS.get(component, component.replace("_", "\n"))


def component_color(component: str) -> str:
    if component == "decomposition_incomplete":
        return "#d62728"
    if component.startswith("l2_loop_baseline"):
        return STAGE_COLORS["control_l2"]
    if component.startswith("dram_loop_baseline"):
        return STAGE_COLORS["control_dram"]
    if component.startswith("l2"):
        return STAGE_COLORS["l2"]
    if component.startswith("dram"):
        return STAGE_COLORS["dram"]
    if component.startswith("compute"):
        return STAGE_COLORS["compute"]
    if component.startswith("hbm"):
        return "#59a14f"
    if component.startswith("gpu_side"):
        return "#f28e2b"
    return "#8c8c8c"


def align16(v: int) -> int:
    return max(16, (int(v) // 16) * 16)


def pj_from_slope(slope_w_per_gbs: float) -> float:
    if not math.isfinite(slope_w_per_gbs):
        return float("nan")
    return slope_w_per_gbs * 1000.0 / 8.0


def stdev(vals: list[float]) -> float:
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def percentile(vals: list[float], p: float) -> float:
    return float(np.percentile(vals, p)) if vals else float("nan")


def finite(v: float) -> bool:
    return math.isfinite(float(v))


def mean(vals: list[float]) -> float:
    return statistics.fmean(vals) if vals else float("nan")


def make_specs(l2_bytes: int, dram_bytes: int, l2_cache_op: str,
               dram_cache_op: str, include_compute: bool) -> list[WorkloadSpec]:
    l2_n = l2_bytes // 16
    dram_n = dram_bytes // 16
    specs = [
        WorkloadSpec("control_l2", "control_l2_read", "decomp_control_read", "none",
                     False, "none", False, l2_n, l2_bytes),
        WorkloadSpec("l2", f"l2_read_{l2_cache_op}", f"decomp_stream_read_{l2_cache_op}", "l2",
                     True, l2_cache_op, False, l2_n, l2_bytes),
        WorkloadSpec("control_dram", "control_dram_read", "decomp_control_read", "none",
                     False, "none", False, dram_n, dram_bytes),
        WorkloadSpec("dram", f"dram_read_{dram_cache_op}", f"decomp_stream_read_{dram_cache_op}", "dram",
                     True, dram_cache_op, False, dram_n, dram_bytes),
    ]
    if include_compute:
        specs.append(
            WorkloadSpec("compute", "compute_fma", "decomp_compute_fma", "none",
                         False, "none", True, dram_n, dram_bytes)
        )
    return specs


def launch(spec: WorkloadSpec, kernels: dict[str, object], stream,
           blocks: int, threads: int, buffers: dict[str, object], sink,
           passes: int) -> None:
    kernel = kernels[spec.kernel_name]
    with stream:
        if spec.is_compute:
            # Compute reference intentionally shares the no-global-memory signature,
            # but dispatch it explicitly so future signature changes do not silently
            # fall through the control-path branch.
            kernel((blocks,), (threads,), (sink, np.uint64(spec.n_u4), np.int32(passes)))
        elif not spec.does_global_read:
            kernel((blocks,), (threads,), (sink, np.uint64(spec.n_u4), np.int32(passes)))
        else:
            if spec.buffer_kind not in buffers:
                raise KeyError(f"missing buffer for stage={spec.stage} buffer_kind={spec.buffer_kind}")
            kernel((blocks,), (threads,),
                   (buffers[spec.buffer_kind], sink, np.uint64(spec.n_u4), np.int32(passes)))


def warm_l2_if_needed(spec: WorkloadSpec, kernels: dict[str, object], stream,
                      blocks: int, threads: int, buffers: dict[str, object], sink,
                      warm_passes: int) -> None:
    if spec.stage != "l2" or warm_passes <= 0:
        return
    launch(spec, kernels, stream, blocks, threads, buffers, sink, warm_passes)
    stream.synchronize()


def calibrate(spec: WorkloadSpec, kernels: dict[str, object], stream,
              blocks: int, threads: int, buffers: dict[str, object], sink,
              cal_passes: int, repeats: int, l2_warm_passes: int) -> tuple[float, float]:
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    best_ms_per_pass = float("inf")
    for _ in range(max(1, repeats)):
        warm_l2_if_needed(spec, kernels, stream, blocks, threads, buffers, sink,
                          l2_warm_passes)
        start.record(stream=stream)
        launch(spec, kernels, stream, blocks, threads, buffers, sink, max(1, cal_passes))
        end.record(stream=stream)
        end.synchronize()
        elapsed = cp.cuda.get_elapsed_time(start, end) / max(1, cal_passes)
        best_ms_per_pass = min(best_ms_per_pass, elapsed)
    passes_per_s = 1.0 / (best_ms_per_pass * 1e-3)
    return best_ms_per_pass, passes_per_s


def run_phase(spec: WorkloadSpec, repeat: int, target: int,
              kernels: dict[str, object], stream, blocks: int, threads: int,
              buffers: dict[str, object], sink, phase_seconds: float,
              window_ms: float, ms_per_pass: float, poller: base.PowerPoller,
              idle_power_w: float, l2_warm_passes: int) -> PhaseRow:
    if target < 0 or target > 100:
        raise ValueError(f"target must be 0..100: {target}")

    phase = f"r{repeat:02d}_{spec.name}_{target:03d}"
    if target == 0:
        passes = 0
        window_s = 0.0
    else:
        active_ms = window_ms * target / 100.0
        passes = max(1, int(round(active_ms / max(ms_per_pass, 1e-9))))
        window_s = window_ms / 1000.0

    warm_l2_if_needed(spec, kernels, stream, blocks, threads, buffers, sink,
                      l2_warm_passes)
    launches = 0

    color = {"control_l2": "gray", "l2": "blue", "control_dram": "gray", "dram": "red"}.get(spec.stage, "green")
    with nvtx.annotate(phase, color=color):
        poller.set_phase(phase)
        t0_abs = time.perf_counter()
        t0 = t0_abs - poller.t0
        deadline = t0_abs + phase_seconds
        if target == 0:
            time.sleep(max(0.0, phase_seconds))
        else:
            while time.perf_counter() < deadline:
                w0 = time.perf_counter()
                launch(spec, kernels, stream, blocks, threads, buffers, sink, passes)
                stream.synchronize()
                launches += 1
                if target < 100:
                    rest = window_s - (time.perf_counter() - w0)
                    if rest > 2e-4:
                        time.sleep(rest)
        t1 = time.perf_counter() - poller.t0

    wall_s = max(t1 - t0, 1e-12)
    nominal_bytes = int(launches * passes * spec.buf_bytes)
    bw = nominal_bytes / wall_s / 1e9
    total_e = poller.energy_j(t0, t1)
    idle_e = idle_power_w * wall_s
    dyn_e = max(0.0, total_e - idle_e)
    avg_p = total_e / wall_s if total_e > 0 else float("nan")
    dyn_p = dyn_e / wall_s
    pj = dyn_e / (nominal_bytes * 8.0) * 1e12 if nominal_bytes > 0 else float("nan")

    samples = poller.slice(t0, t1)
    power = [s.power_w for s in samples if s.power_w >= 0]
    instant = [s.power_instant_w for s in samples if s.power_instant_w >= 0]
    average = [s.power_average_w for s in samples if s.power_average_w >= 0]
    sm_clocks = [s.sm_clock_mhz for s in samples if s.sm_clock_mhz >= 0]
    mem_clocks = [s.mem_clock_mhz for s in samples if s.mem_clock_mhz >= 0]
    temps = [s.temp_gpu_c for s in samples if s.temp_gpu_c >= 0]
    pstates = [s.pstate for s in samples if s.pstate >= 0]
    gpu_utils = [s.gpu_util_pct for s in samples if s.gpu_util_pct >= 0]
    mem_utils = [s.mem_util_pct for s in samples if s.mem_util_pct >= 0]

    return PhaseRow(
        repeat=repeat,
        stage=spec.stage,
        target_pct=target,
        phase=phase,
        t0_s=t0,
        t1_s=t1,
        wall_s=wall_s,
        launches=launches,
        passes_per_launch=passes,
        nominal_bytes=nominal_bytes,
        nominal_bandwidth_gbs=bw,
        total_energy_j=total_e,
        idle_energy_j=idle_e,
        dynamic_energy_j=dyn_e,
        avg_power_w=avg_p,
        dynamic_power_w=dyn_p,
        pj_per_nominal_bit=pj,
        samples=len(samples),
        power_std_w=stdev(power),
        power_p05_w=percentile(power, 5),
        power_p50_w=percentile(power, 50),
        power_p95_w=percentile(power, 95),
        sm_clock_mhz_mean=mean(sm_clocks),
        mem_clock_mhz_mean=mean(mem_clocks),
        temp_gpu_c_mean=mean(temps),
        pstate_p50=percentile(pstates, 50),
        gpu_util_pct_mean=mean(gpu_utils),
        mem_util_pct_mean=mean(mem_utils),
        avg_power_instant_w=mean(instant),
        avg_power_average_w=mean(average),
        power_instant_samples=len(instant),
        power_average_samples=len(average),
    )


def make_aggregate_rows(rows: list[PhaseRow], min_repeat: int) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    stages = sorted({r.stage for r in rows})
    for stage in stages:
        targets = sorted({r.target_pct for r in rows if r.stage == stage})
        for target in targets:
            group = [
                r for r in rows
                if r.stage == stage and r.target_pct == target and r.repeat >= min_repeat
                and r.nominal_bandwidth_gbs > 0 and finite(r.avg_power_w)
            ]
            if not group:
                continue
            powers = [r.avg_power_w for r in group]
            bws = [r.nominal_bandwidth_gbs for r in group]
            out.append({
                "stage": stage,
                "target_pct": target,
                "n": len(group),
                "repeat_min_used": min(r.repeat for r in group),
                "repeat_max_used": max(r.repeat for r in group),
                "bandwidth_gbs_median": float(np.median(bws)),
                "bandwidth_gbs_iqr": float(np.percentile(bws, 75) - np.percentile(bws, 25)),
                "avg_power_w_median": float(np.median(powers)),
                "avg_power_w_iqr": float(np.percentile(powers, 75) - np.percentile(powers, 25)),
                "avg_power_w_mean": float(np.mean(powers)),
                "avg_power_w_std": float(np.std(powers)),
            })
    return out


def fit_power_vs_bw(stage: str, rows: list[PhaseRow], min_target: int,
                    min_repeat: int, aggregate: str) -> FitResult | None:
    filtered = [
        r for r in rows
        if r.stage == stage
        and r.repeat >= min_repeat
        and r.target_pct >= min_target
        and r.nominal_bandwidth_gbs > 0
        and finite(r.avg_power_w)
    ]
    if len(filtered) < 2:
        return None

    if aggregate == "median":
        agg = [r for r in make_aggregate_rows(filtered, min_repeat)
               if r["stage"] == stage and int(r["target_pct"]) >= min_target]
        if len(agg) < 2:
            return None
        xs = np.array([float(r["bandwidth_gbs_median"]) for r in agg], dtype=np.float64)
        ys = np.array([float(r["avg_power_w_median"]) for r in agg], dtype=np.float64)
        targets = ",".join(str(int(r["target_pct"])) for r in agg)
        n_points = len(agg)
    elif aggregate == "raw":
        xs = np.array([r.nominal_bandwidth_gbs for r in filtered], dtype=np.float64)
        ys = np.array([r.avg_power_w for r in filtered], dtype=np.float64)
        targets = ",".join(str(t) for t in sorted({r.target_pct for r in filtered}))
        n_points = len(filtered)
    else:
        raise ValueError(f"unknown aggregate mode: {aggregate}")

    # Ordinary least squares for y = slope*x + intercept. With median aggregation,
    # this is a target-level robust fit rather than a phase-level noise fit.
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
    pred = slope * xs + intercept
    resid = ys - pred
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return FitResult(
        stage=stage,
        n_points=n_points,
        targets=targets,
        slope_w_per_gbs=float(slope),
        intercept_w=float(intercept),
        r2=float(r2),
        max_abs_residual_w=float(np.max(np.abs(resid))) if len(resid) else float("nan"),
        pj_per_nominal_bit=pj_from_slope(float(slope)),
    )


def build_decomp(fits: dict[str, FitResult], hbm_pjbit: float | None,
                 l2_cache_op: str, dram_cache_op: str) -> list[DecompResult]:
    def pj(stage: str) -> float:
        return fits[stage].pj_per_nominal_bit if stage in fits else float("nan")

    def single_stage_note(stage: str) -> str:
        if stage == "control_l2":
            return "Partial run: matched L2-sized no-read baseline loop slope only; not a physical memory path."
        if stage == "l2":
            return f"Partial run: L2-resident ld.global.{l2_cache_op} stage slope; full L2 read increment requires control_l2 too."
        if stage == "control_dram":
            return "Partial run: matched DRAM-sized no-read baseline loop slope only; not a physical memory path."
        if stage == "dram":
            return f"Partial run: DRAM-streaming ld.global.{dram_cache_op} stage slope; full off-chip decomposition requires all four main stages."
        if stage == "compute":
            return "Partial run: no-global-memory FMA reference normalized by nominal loop bytes; not memory pJ/bit."
        return "Partial run stage slope."

    missing_required = [stage for stage in REQUIRED_DECOMP_STAGES if stage not in fits]
    if missing_required:
        out: list[DecompResult] = []
        for stage in STAGE_ORDER:
            if stage in fits:
                out.append(DecompResult(
                    f"{stage}_stage_slope",
                    pj(stage),
                    single_stage_note(stage),
                ))
        out.append(DecompResult(
            "decomposition_incomplete",
            float("nan"),
            "Complete lower-bound decomposition requires control_l2,l2,control_dram,dram fits; "
            f"missing {','.join(missing_required)}. This is expected for --only-stage / NCU validation runs.",
        ))
        return out

    ctrl_l2 = pj("control_l2")
    l2 = pj("l2")
    ctrl_dram = pj("control_dram")
    dram = pj("dram")
    l2_path = l2 - ctrl_l2 if finite(l2) and finite(ctrl_l2) else float("nan")
    dram_path = dram - ctrl_dram if finite(dram) and finite(ctrl_dram) else float("nan")
    offchip_increment = dram_path - l2_path if finite(dram_path) and finite(l2_path) else float("nan")

    out = [
        DecompResult(
            "l2_loop_baseline",
            ctrl_l2,
            "Matched no-read index/loop/accumulator cost per L2-buffer nominal bit; not a physical memory path.",
            "control_l2_loop",
        ),
        DecompResult(
            "l2_read_total",
            l2,
            f"Board-level slope for L2-resident ld.global.{l2_cache_op} loads, including SM/RF/LSU/L2 and active clocks.",
        ),
        DecompResult(
            "dram_loop_baseline",
            ctrl_dram,
            "Matched no-read index/loop/accumulator cost per DRAM-buffer nominal bit; not a physical memory path.",
            "control_dram_loop",
        ),
        DecompResult(
            "dram_read_total",
            dram,
            f"Board-level slope for DRAM-streaming ld.global.{dram_cache_op} loads over a buffer much larger than L2.",
        ),
        DecompResult(
            "l2_read_increment",
            l2_path,
            "Lower-bound SM/RF/LSU/L2 read-path increment above the L2-sized no-read baseline loop.",
            "l2_over_control",
        ),
        DecompResult(
            "dram_read_increment_total",
            dram_path,
            "Whole DRAM-stream global-read path above the DRAM-sized no-read baseline loop; still board-level.",
            "dram_over_control",
        ),
        DecompResult(
            "post_l2_offchip_increment",
            offchip_increment,
            "Additional miss/off-chip increment after removing the L2 read-path estimate: MC + GPU PHY + HBM PHY/core.",
            "dram_global_over_l2",
        ),
    ]
    if "compute" in fits:
        out.append(DecompResult(
            "compute_only_reference",
            pj("compute"),
            "No-global-memory FMA reference normalized by nominal loop bytes; use as SM-dynamic sanity, not memory pJ/bit.",
        ))
    if hbm_pjbit is not None and math.isfinite(hbm_pjbit):
        out.extend([
            DecompResult(
                "hbm_stack_external_prior",
                hbm_pjbit,
                "External HBM stack/PHY prior supplied by --hbm-pjbit; not measured by NVML.",
            ),
            DecompResult(
                "gpu_side_after_l2_est",
                offchip_increment - hbm_pjbit if finite(offchip_increment) else float("nan"),
                "Post-L2 off-chip increment minus external HBM prior; estimates MC/GPU-PHY side residual.",
            ),
            DecompResult(
                "gpu_side_whole_read_path_est",
                dram_path - hbm_pjbit if finite(dram_path) else float("nan"),
                "DRAM read increment total minus external HBM prior; estimates SM-to-GPU-PHY side residual.",
            ),
        ])
    return out

def save_rows(path: Path, rows: Iterable[object | dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    if isinstance(rows[0], dict):
        dict_rows = rows  # type: ignore[assignment]
    else:
        dict_rows = [vars(r) for r in rows]
    fieldnames: list[str] = []
    for row in dict_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(dict_rows)


def save_trace(path: Path, samples: list[base.PowerSample]) -> None:
    fields = [
        "t_s", "power_w", "power_instant_w", "power_average_w",
        "power_instant_status", "power_average_status",
        "gpu_util_pct", "mem_util_pct", "sm_clock_mhz", "mem_clock_mhz",
        "temp_gpu_c", "pstate", "throttle_reasons", "phase",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for s in samples:
            d = vars(s).copy()
            d["throttle_reasons"] = f"0x{s.throttle_reasons:x}" if s.throttle_reasons >= 0 else ""
            writer.writerow({k: d.get(k, "") for k in fields})



def save_power_plot(path: Path, gpu_name: str, rows: list[PhaseRow],
                    samples: list[base.PowerSample], idle_power_w: float) -> None:
    """Telemetry sanity plot.  Slope interpretation is in save_fit_plot()."""
    if not samples:
        return
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16, 9), sharex=False)
    good = [s for s in samples if s.power_w >= 0]
    ax0.plot([s.t_s for s in good], [s.power_w for s in good], lw=0.8,
             label="NVML power")
    ax0.axhline(idle_power_w, ls="--", lw=1.0, label=f"pre-idle {idle_power_w:.1f} W")
    ymax = max([s.power_w for s in good], default=idle_power_w)
    label_stride = max(1, math.ceil(len(rows) / 24)) if rows else 1
    for idx, r in enumerate(rows):
        ax0.axvspan(r.t0_s, r.t1_s, alpha=0.08, color=stage_color(r.stage))
        if len(rows) <= 32 or idx % label_stride == 0:
            ax0.text((r.t0_s + r.t1_s) / 2, ymax, f"r{r.repeat}:{r.stage}:{r.target_pct}",
                     ha="center", va="bottom", rotation=75, fontsize=7)
    ax0.set_title(f"Power trace - {gpu_name}")
    ax0.set_ylabel("power (W)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left", fontsize=8)

    for stage in STAGE_ORDER:
        rs = [r for r in rows if r.stage == stage and r.target_pct > 0]
        if not rs:
            continue
        ax1.scatter([r.nominal_bandwidth_gbs for r in rs], [r.avg_power_w for r in rs],
                    label=stage, alpha=0.45, s=16, color=stage_color(stage))
    ax1.set_title("Raw active phases: average power vs nominal bandwidth")
    ax1.set_xlabel("nominal bandwidth (GB/s; control phases use loop-equivalent nominal bytes)")
    ax1.set_ylabel("average power (W)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def safe_bar_value(v: float) -> float:
    return float(v) if finite(v) else 0.0


def safe_bar_label(v: float, fmt: str = ".3f") -> str:
    return format(float(v), fmt) if finite(v) else "n/a"


def save_fit_plot(path: Path, gpu_name: str, rows: list[PhaseRow],
                  aggregate_rows: list[dict[str, float | int | str]],
                  fits: list[FitResult], min_repeat: int,
                  fit_min_target: int, fit_aggregate: str) -> None:
    """Primary estimator plot: raw repeats, median/IQR target points, fit lines."""
    if not rows:
        return
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16, 10), sharex=False)
    fit_by_stage = {f.stage: f for f in fits}
    agg_by_stage: dict[str, list[dict[str, float | int | str]]] = {}
    for row in aggregate_rows:
        agg_by_stage.setdefault(str(row["stage"]), []).append(row)

    for stage in STAGE_ORDER:
        raw = [r for r in rows if r.stage == stage and r.repeat >= min_repeat
               and r.target_pct >= fit_min_target and r.nominal_bandwidth_gbs > 0
               and finite(r.avg_power_w)]
        if not raw and stage not in fit_by_stage:
            continue
        color = stage_color(stage)
        if raw:
            ax0.scatter([r.nominal_bandwidth_gbs for r in raw], [r.avg_power_w for r in raw],
                        s=14, alpha=0.18, color=color)
        agg = sorted([r for r in agg_by_stage.get(stage, [])
                      if int(r["target_pct"]) >= fit_min_target and float(r["bandwidth_gbs_median"]) > 0],
                     key=lambda r: int(r["target_pct"]))
        if agg:
            xs = np.array([float(r["bandwidth_gbs_median"]) for r in agg], dtype=np.float64)
            ys = np.array([float(r["avg_power_w_median"]) for r in agg], dtype=np.float64)
            xerr = np.array([float(r.get("bandwidth_gbs_iqr", 0.0)) / 2.0 for r in agg], dtype=np.float64)
            yerr = np.array([float(r.get("avg_power_w_iqr", 0.0)) / 2.0 for r in agg], dtype=np.float64)
            ax0.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt="o", capsize=3,
                         color=color, ecolor=color, markersize=6)
            for x, y, r in zip(xs, ys, agg):
                ax0.annotate(str(int(r["target_pct"])), (x, y), textcoords="offset points",
                             xytext=(5, 5), fontsize=8, color=color)
        fit = fit_by_stage.get(stage)
        if fit is not None:
            source_x = [r.nominal_bandwidth_gbs for r in raw]
            if agg:
                source_x += [float(r["bandwidth_gbs_median"]) for r in agg]
            if source_x:
                xmin, xmax = min(source_x), max(source_x)
                if xmax <= xmin:
                    xmax = xmin + 1e-9
                line_x = np.linspace(xmin, xmax, 80)
                ax0.plot(line_x, fit.slope_w_per_gbs * line_x + fit.intercept_w,
                         color=color, lw=2.0,
                         label=f"{stage}: {fit.pj_per_nominal_bit:.2f} pJ/bit R2={safe_bar_label(fit.r2)} n={fit.n_points}")

        residuals_x: list[int] = []
        residuals_y: list[float] = []
        if fit is not None:
            for r in agg:
                x = float(r["bandwidth_gbs_median"])
                y = float(r["avg_power_w_median"])
                residuals_x.append(int(r["target_pct"]))
                residuals_y.append(y - (fit.slope_w_per_gbs * x + fit.intercept_w))
        if residuals_y:
            ax1.plot(residuals_x, residuals_y, marker="o", color=color, label=stage)

    ax0.set_title(f"Slope estimator evidence - {gpu_name}")
    ax0.set_xlabel("nominal bandwidth (GB/s)")
    ax0.set_ylabel("average power (W)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=8, loc="best")
    ax0.text(0.01, 0.01, f"Fit mode={fit_aggregate}; faint dots=raw repeats; error bars=target median IQR/2.",
             transform=ax0.transAxes, ha="left", va="bottom", fontsize=8,
             bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.8", "alpha": 0.9})
    ax1.axhline(0.0, color="black", lw=0.8, ls="--")
    ax1.set_title("Fit residuals from target median points")
    ax1.set_xlabel("target (%)")
    ax1.set_ylabel("residual power (W)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_diagnostics_plot(path: Path, gpu_name: str, rows: list[PhaseRow], args) -> None:
    """Repeat-level drift plot for power, bandwidth, clock, and temperature."""
    active = [r for r in rows if r.repeat >= args.warmup_repeats and r.target_pct > 0]
    if not active:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax0, ax1, ax2, ax3 = axes.ravel()
    for stage in STAGE_ORDER:
        rs = [r for r in active if r.stage == stage]
        if not rs:
            continue
        c = stage_color(stage)
        ax0.scatter([r.repeat for r in rs], [r.avg_power_w for r in rs], s=14, alpha=0.35, color=c, label=stage)
        ax1.scatter([r.repeat for r in rs], [r.nominal_bandwidth_gbs for r in rs], s=14, alpha=0.35, color=c, label=stage)
        ax2.scatter([r.repeat for r in rs], [r.sm_clock_mhz_mean for r in rs], s=14, alpha=0.35, color=c, label=stage)
        ax3.scatter([r.repeat for r in rs], [r.temp_gpu_c_mean for r in rs], s=14, alpha=0.35, color=c, label=stage)
    ax0.set_title("Average power by repeat")
    ax0.set_ylabel("average power (W)")
    ax1.set_title("Nominal bandwidth by repeat")
    ax1.set_ylabel("GB/s")
    ax2.set_title("SM clock by repeat")
    ax2.set_ylabel("MHz")
    ax3.set_title("GPU temperature by repeat")
    ax3.set_ylabel("C")
    ax3.axhline(args.temperature_warn_c, color="black", ls="--", lw=1.0, label=f"warn {args.temperature_warn_c:.0f} C")
    for ax in axes.ravel():
        ax.set_xlabel("repeat")
        ax.grid(True, alpha=0.3)
    ax0.legend(fontsize=8)
    ax3.legend(fontsize=8)
    fig.suptitle(f"Run diagnostics - {gpu_name}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_method_plot(path: Path, gpu_name: str, decomp: list[DecompResult],
                     l2_cache_op: str, dram_cache_op: str, hbm_pjbit: float | None) -> None:
    """Static diagram of the decomposition equations."""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")

    def box(x: float, y: float, text: str, color: str, w: float = 0.18, h: float = 0.12) -> None:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", alpha=0.82))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    def arrow(x0: float, y0: float, x1: float, y1: float, label: str = "") -> None:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 1.4})
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.025, label, ha="center", va="bottom", fontsize=9)

    box(0.05, 0.68, "L2 loop\nbaseline", stage_color("control_l2"))
    box(0.05, 0.46, f"l2\nld.global.{l2_cache_op}", stage_color("l2"))
    box(0.33, 0.56, "l2_read_increment\n= l2_read_total\n- l2_loop_baseline", component_color("l2_read_increment"), w=0.24, h=0.16)
    arrow(0.23, 0.74, 0.33, 0.64)
    arrow(0.23, 0.52, 0.33, 0.60)
    box(0.05, 0.26, "DRAM loop\nbaseline", stage_color("control_dram"))
    box(0.05, 0.04, f"dram\nld.global.{dram_cache_op}", stage_color("dram"))
    box(0.33, 0.14, "dram_read_increment_total\n= dram_read_total\n- dram_loop_baseline", component_color("dram_read_increment_total"), w=0.24, h=0.16)
    arrow(0.23, 0.32, 0.33, 0.22)
    arrow(0.23, 0.10, 0.33, 0.18)
    box(0.65, 0.35, "post_l2_offchip_increment\n= dram_read_increment_total\n- l2_read_increment", component_color("post_l2_offchip_increment"), w=0.26, h=0.18)
    arrow(0.57, 0.62, 0.65, 0.47, "subtract")
    arrow(0.57, 0.20, 0.65, 0.40, "subtract")
    if hbm_pjbit is not None and math.isfinite(hbm_pjbit):
        box(0.65, 0.08, f"external HBM prior\n{hbm_pjbit:.3f} pJ/bit", component_color("hbm_stack_external_prior"), w=0.26)
        box(0.83, 0.22, "GPU-side residual\n= off-chip increment\n- HBM prior", component_color("gpu_side_after_l2_est"), w=0.16, h=0.17)
        arrow(0.78, 0.35, 0.83, 0.31)
        arrow(0.78, 0.14, 0.83, 0.27)
    else:
        ax.text(0.64, 0.12, "No --hbm-pjbit supplied:\nHBM PHY/core is not separated.\nMeasurable value = MC + GPU PHY + HBM PHY/core.",
                ha="left", va="center", fontsize=10,
                bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "0.65"})
    if any(d.component == "decomposition_incomplete" for d in decomp):
        ax.text(0.50, 0.88, "Partial validation run: complete decomposition requires all four main stages.",
                ha="center", va="center", fontsize=11, color="darkred",
                bbox={"boxstyle": "round,pad=0.35", "fc": "#fff5f5", "ec": "darkred"})
    ax.text(0.02, 0.94, f"Read-path decomposition logic - {gpu_name}", ha="left", va="center", fontsize=15, weight="bold")
    ax.text(0.02, 0.90, "Slopes are board/NVML-level avg_power~nominal_bandwidth fits; baseline slopes are no-read loop-equivalent baselines.",
            ha="left", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_decomp_plot(path: Path, gpu_name: str, fits: list[FitResult],
                     decomp: list[DecompResult]) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(17, 6))
    fit_labels = [stage_label(f.stage) for f in fits]
    fit_vals = [safe_bar_value(f.pj_per_nominal_bit) for f in fits]
    x0 = np.arange(len(fit_labels))
    ax0.bar(x0, fit_vals, color=[stage_color(f.stage) for f in fits])
    ax0.set_title("stage slope pJ / nominal bit")
    ax0.set_ylabel("pJ/bit")
    ax0.set_xticks(x0, fit_labels)
    ax0.grid(axis="y", alpha=0.3)
    for i, f in enumerate(fits):
        ax0.text(i, safe_bar_value(f.pj_per_nominal_bit),
                 f"{safe_bar_label(f.pj_per_nominal_bit)}\nR2={safe_bar_label(f.r2)}\nn={f.n_points}",
                 ha="center", va="bottom", fontsize=8)
    labels = [component_label(d.component) for d in decomp]
    vals = [safe_bar_value(d.pj_per_bit) for d in decomp]
    x1 = np.arange(len(labels))
    ax1.bar(x1, vals, color=[component_color(d.component) for d in decomp])
    ax1.set_title("derived lower-bound components")
    ax1.set_ylabel("pJ/bit")
    ax1.set_xticks(x1, labels, rotation=45, ha="right")
    ax1.axhline(0.0, color="black", lw=0.8)
    ax1.grid(axis="y", alpha=0.3)
    for i, d in enumerate(decomp):
        if finite(d.pj_per_bit):
            ax1.text(i, vals[i], safe_bar_label(d.pj_per_bit), ha="center", va="bottom", fontsize=8)
        else:
            ax1.text(i, 0.0, "n/a", ha="center", va="bottom", fontsize=8, rotation=90)
    fig.suptitle(f"Read-path energy decomposition - {gpu_name}")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def quality_checks(rows: list[PhaseRow], fits: list[FitResult], args, idle_stats: dict[str, float | int | str], env_actions: list[dict[str, object]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []

    def add(level: str, check: str, detail: str) -> None:
        out.append({"level": level, "check": check, "detail": detail})

    if args.repeats <= args.warmup_repeats:
        add("warn", "repeat_coverage", "--repeats must be greater than --warmup-repeats for a non-empty fit set")
    elif args.repeats - args.warmup_repeats < 10:
        add("warn", "repeat_coverage", f"only {args.repeats - args.warmup_repeats} fit repeats after warmup; median/IQR stability may be weak")
    else:
        add("ok", "repeat_coverage", f"{args.repeats - args.warmup_repeats} fit repeats after warmup")

    idle_iqr = float(idle_stats.get("iqr_w", 0.0))
    if idle_iqr > args.idle_iqr_warn_w:
        add("warn", "idle_iqr", f"idle power IQR {idle_iqr:.3f} W exceeds threshold {args.idle_iqr_warn_w:.3f} W")
    else:
        add("ok", "idle_iqr", f"idle power IQR {idle_iqr:.3f} W")

    if args.idle_filter_sm_clock_below > 0:
        filtered_samples = int(idle_stats.get("filtered_samples", 0))
        if str(idle_stats.get("baseline_source", "")) == "all_idle_samples" and filtered_samples < args.idle_filter_min_samples:
            add("warn", "idle_filter",
                f"only {filtered_samples} idle samples below {args.idle_filter_sm_clock_below} MHz; baseline fell back to all samples")
        else:
            add("ok", "idle_filter",
                f"idle baseline source={idle_stats.get('baseline_source', '')}, filtered_samples={filtered_samples}")

    failed_env = [a for a in env_actions if int(a.get("returncode", 0)) != 0]
    if failed_env:
        add("warn", "gpu_setup", f"{len(failed_env)} nvidia-smi setup command(s) failed; see metadata env_actions")
    elif env_actions:
        add("ok", "gpu_setup", f"{len(env_actions)} nvidia-smi setup command(s) succeeded")
    else:
        add("warn", "gpu_setup", "no clock/persistence/power-limit setup requested; compare clocks in quality CSV")

    present_stages = {r.stage for r in rows}
    fit_stages = {f.stage for f in fits}
    missing_row_stages = [stage for stage in REQUIRED_DECOMP_STAGES if stage not in present_stages]
    missing_fit_stages = [stage for stage in REQUIRED_DECOMP_STAGES if stage not in fit_stages]
    if args.only_stage:
        add("warn", "partial_decomposition",
            f"--only-stage={args.only_stage} is a validation/NCU mode; complete decomposition requires {','.join(REQUIRED_DECOMP_STAGES)}")
    elif missing_row_stages:
        add("warn", "decomposition_stage_coverage",
            f"missing measured stage(s): {','.join(missing_row_stages)}")
    elif missing_fit_stages:
        add("warn", "decomposition_fit_coverage",
            f"all stages were measured but fit(s) are missing: {','.join(missing_fit_stages)}")
    else:
        add("ok", "decomposition_coverage", "all four main stages have fits for complete decomposition")

    if args.l2_warm_passes < 4:
        add("warn", "l2_warm_passes",
            f"--l2-warm-passes={args.l2_warm_passes}; use >=4 for stronger L2 residency before l2 phases")
    else:
        add("ok", "l2_warm_passes", f"--l2-warm-passes={args.l2_warm_passes}")

    for stage in sorted({r.stage for r in rows}):
        stage_rows = [r for r in rows if r.stage == stage]
        if not stage_rows:
            add("warn", f"{stage}_coverage", "no rows")
            continue
        low_sample = [r.phase for r in stage_rows if r.samples < 3]
        if low_sample:
            add("warn", f"{stage}_samples", f"low NVML sample count: {low_sample[:8]}")
        else:
            add("ok", f"{stage}_samples", "all phases have >=3 samples")
        active = [r for r in stage_rows if r.repeat >= args.warmup_repeats and r.target_pct >= args.fit_min_target]
        pstates = sorted({int(round(r.pstate_p50)) for r in active if finite(r.pstate_p50)})
        if len(pstates) > 1:
            add("warn", f"{stage}_pstate", f"active phases use multiple P-states: {pstates}")
        elif pstates:
            add("ok", f"{stage}_pstate", f"active phases P-state P{pstates[0]}")
        sm_clocks = [r.sm_clock_mhz_mean for r in active if finite(r.sm_clock_mhz_mean)]
        mem_clocks = [r.mem_clock_mhz_mean for r in active if finite(r.mem_clock_mhz_mean)]
        if sm_clocks and (max(sm_clocks) - min(sm_clocks) > args.clock_warn_mhz):
            add("warn", f"{stage}_sm_clock", f"SM clock spread {max(sm_clocks)-min(sm_clocks):.1f} MHz")
        if mem_clocks and (max(mem_clocks) - min(mem_clocks) > args.clock_warn_mhz):
            add("warn", f"{stage}_mem_clock", f"memory clock spread {max(mem_clocks)-min(mem_clocks):.1f} MHz")
        temps = [r.temp_gpu_c_mean for r in active if finite(r.temp_gpu_c_mean)]
        if temps:
            max_temp = max(temps)
            if max_temp > args.temperature_warn_c:
                add("warn", f"{stage}_temperature",
                    f"max active temperature {max_temp:.1f} C exceeds threshold {args.temperature_warn_c:.1f} C")
            else:
                add("ok", f"{stage}_temperature", f"max active temperature {max_temp:.1f} C")

    for fit in fits:
        if fit.n_points < 3:
            add("warn", f"{fit.stage}_fit_points",
                f"only {fit.n_points} fit point(s); two-point linear fits have trivially perfect R2")
        elif fit.n_points < len([t for t in args.targets if t >= args.fit_min_target]):
            add("warn", f"{fit.stage}_fit_points",
                f"fit used {fit.n_points} point(s), fewer than requested active targets >= {args.fit_min_target}")
        else:
            add("ok", f"{fit.stage}_fit_points", f"fit used {fit.n_points} point(s)")

        if not finite(fit.r2):
            add("warn", f"{fit.stage}_fit",
                f"R2 is not finite, max residual={fit.max_abs_residual_w:.3f} W")
        elif fit.r2 < args.r2_warn:
            add("warn", f"{fit.stage}_fit", f"R2={fit.r2:.6f}, max residual={fit.max_abs_residual_w:.3f} W")
        else:
            add("ok", f"{fit.stage}_fit", f"R2={fit.r2:.6f}, max residual={fit.max_abs_residual_w:.3f} W")

    if args.dram_buf_bytes < args.l2_size_bytes * args.min_dram_l2_multiple:
        add("warn", "dram_buffer_size", f"DRAM buffer is less than {args.min_dram_l2_multiple}x L2; L2 miss assumption is weaker")
    else:
        add("ok", "dram_buffer_size", f"DRAM buffer is >= {args.min_dram_l2_multiple}x L2")
    if args.l2_buf_bytes > args.l2_size_bytes * 0.50:
        add("warn", "l2_buffer_size", "L2 buffer is >50% of L2; residency may be fragile")
    else:
        add("ok", "l2_buffer_size", "L2 buffer is <=50% of L2")
    return out



def run_cmd(cmd: list[str], label: str) -> dict[str, object]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        return {
            "label": label,
            "cmd": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"label": label, "cmd": " ".join(cmd), "returncode": -1, "stderr": str(exc)}


def configure_gpu_with_nvidia_smi(args) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi is None:
        if any([args.set_persistence, args.lock_gpu_clocks, args.lock_memory_clocks,
                args.lock_memory_clocks_deferred, args.power_limit_w is not None,
                args.reset_gpu_clocks_before, args.reset_memory_clocks_before]):
            actions.append({"label": "nvidia-smi", "returncode": -1,
                            "stderr": "nvidia-smi not found"})
        return actions

    dev = str(args.device)
    if args.set_persistence:
        actions.append(run_cmd([nvsmi, "-i", dev, "-pm", "1"], "set_persistence_mode"))
    if args.reset_gpu_clocks_before:
        actions.append(run_cmd([nvsmi, "-i", dev, "-rgc"], "reset_gpu_clocks_before"))
    if args.reset_memory_clocks_before:
        actions.append(run_cmd([nvsmi, "-i", dev, "-rmc"], "reset_memory_clocks_before"))
    if args.power_limit_w is not None:
        cmd = [nvsmi, "-i", dev, "-pl", str(args.power_limit_w)]
        if args.power_limit_scope is not None:
            cmd += ["--scope", str(args.power_limit_scope)]
        actions.append(run_cmd(cmd, "set_power_limit"))
    if args.lock_gpu_clocks:
        actions.append(run_cmd([nvsmi, "-i", dev, "-lgc", args.lock_gpu_clocks],
                               "lock_gpu_clocks"))
    if args.lock_memory_clocks:
        actions.append(run_cmd([nvsmi, "-i", dev, "-lmc", args.lock_memory_clocks],
                               "lock_memory_clocks"))
    if args.lock_memory_clocks_deferred:
        actions.append(run_cmd([nvsmi, "-i", dev, "-lmcd", args.lock_memory_clocks_deferred],
                               "lock_memory_clocks_deferred"))
    return actions


def cleanup_gpu_with_nvidia_smi(args) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi is None or not args.reset_clocks_on_exit:
        return actions
    dev = str(args.device)
    actions.append(run_cmd([nvsmi, "-i", dev, "-rgc"], "reset_gpu_clocks_on_exit"))
    actions.append(run_cmd([nvsmi, "-i", dev, "-rmc"], "reset_memory_clocks_on_exit"))
    return actions


def summarize_power_values(vals: list[float], prefix: str = "") -> dict[str, float | int]:
    if not vals:
        return {
            f"{prefix}mean_w": 0.0,
            f"{prefix}std_w": 0.0,
            f"{prefix}median_w": 0.0,
            f"{prefix}iqr_w": 0.0,
            f"{prefix}samples": 0,
        }
    arr = np.array(vals, dtype=np.float64)
    return {
        f"{prefix}mean_w": float(np.mean(arr)),
        f"{prefix}std_w": float(np.std(arr)),
        f"{prefix}median_w": float(np.median(arr)),
        f"{prefix}iqr_w": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        f"{prefix}samples": int(len(vals)),
    }


def measure_idle_power_stats(handle, seconds: float, hz: int,
                             sm_clock_filter_mhz: int = 0,
                             min_filtered_samples: int = 5) -> dict[str, float | int | str]:
    interval_s = 1.0 / hz
    deadline = time.perf_counter() + max(0.0, seconds)
    vals: list[float] = []
    filtered_vals: list[float] = []
    sm_clocks: list[int] = []
    pstates: list[int] = []
    while time.perf_counter() < deadline:
        power_mw = base.nvml_or(pynvml.nvmlDeviceGetPowerUsage, handle, default=-1)
        sm_clock = base.nvml_or(
            pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_SM, default=-1)
        pstate = base.nvml_or(pynvml.nvmlDeviceGetPerformanceState, handle, default=-1)
        if sm_clock >= 0:
            sm_clocks.append(int(sm_clock))
        if pstate >= 0:
            pstates.append(int(pstate))
        if power_mw >= 0:
            power_w = power_mw / 1000.0
            vals.append(power_w)
            if sm_clock_filter_mhz <= 0 or (sm_clock >= 0 and sm_clock < sm_clock_filter_mhz):
                filtered_vals.append(power_w)
        sleep_s = min(interval_s, max(0.0, deadline - time.perf_counter()))
        if sleep_s > 0:
            time.sleep(sleep_s)

    all_stats = summarize_power_values(vals)
    filt_stats = summarize_power_values(filtered_vals, "filtered_")
    use_filtered = sm_clock_filter_mhz > 0 and int(filt_stats["filtered_samples"]) >= min_filtered_samples
    out: dict[str, float | int | str] = {**all_stats, **filt_stats}
    if use_filtered:
        out.update({
            "mean_w": float(filt_stats["filtered_mean_w"]),
            "std_w": float(filt_stats["filtered_std_w"]),
            "median_w": float(filt_stats["filtered_median_w"]),
            "iqr_w": float(filt_stats["filtered_iqr_w"]),
            "samples": int(filt_stats["filtered_samples"]),
            "baseline_source": f"sm_clock_filtered_lt_{sm_clock_filter_mhz}_mhz",
        })
    else:
        out["baseline_source"] = "all_idle_samples"
    out["all_samples"] = int(len(vals))
    out["sm_clock_filter_mhz"] = int(sm_clock_filter_mhz)
    out["min_filtered_samples"] = int(min_filtered_samples)
    out["sm_clock_mhz_min"] = min(sm_clocks) if sm_clocks else -1
    out["sm_clock_mhz_max"] = max(sm_clocks) if sm_clocks else -1
    out["pstates_seen"] = ",".join(f"P{p}" for p in sorted(set(pstates))) if pstates else ""
    return out

def print_table(fits: list[FitResult], decomp: list[DecompResult], quality: list[dict[str, str]]) -> None:
    print()
    print(f"{'stage':<10} {'n':>4} {'targets':>12} {'slope W/GB/s':>14} {'intercept W':>12} {'R2':>8} {'pJ/bit':>10}")
    print("-" * 82)
    for f in fits:
        print(f"{f.stage:<10} {f.n_points:>4d} {f.targets:>12} {f.slope_w_per_gbs:>14.6f} "
              f"{f.intercept_w:>12.3f} {f.r2:>8.4f} {f.pj_per_nominal_bit:>10.3f}")
    print()
    print(f"{'component':<32} {'pJ/bit':>12}  note")
    print("-" * 110)
    for d in decomp:
        val = base.fmt_or_na(d.pj_per_bit)
        print(f"{d.component:<32} {val:>12}  {d.note}")
    warns = [q for q in quality if q["level"] != "ok"]
    if warns:
        print()
        print("quality warnings")
        print("-" * 80)
        for q in warns:
            print(f"{q['level']}: {q['check']} - {q['detail']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--targets", type=int, nargs="+", default=[50, 75, 100],
                    help="active duty targets. Recommended: 50 75 100; use 0 only for diagnostics")
    ap.add_argument("--repeats", type=int, default=35)
    ap.add_argument("--phase-seconds", type=float, default=15.0)
    ap.add_argument("--idle-seconds", type=float, default=15.0)
    ap.add_argument("--idle-settle-seconds", type=float, default=5.0,
                    help="extra quiet time before idle baseline to reduce P0/P-state hysteresis")
    ap.add_argument("--idle-filter-sm-clock-below", type=int, default=500,
                    help="prefer idle samples with SM clock below this MHz; 0 disables filtering")
    ap.add_argument("--idle-filter-min-samples", type=int, default=5,
                    help="minimum filtered idle samples before replacing the all-sample baseline")
    ap.add_argument("--gap-seconds", type=float, default=1.0)
    ap.add_argument("--warmup-repeats", type=int, default=5,
                    help="discard the first N repeats from slope fits; raw rows are still saved")
    ap.add_argument("--window-ms", type=float, default=1000.0,
                    help="large default because many datacenter GPUs expose 1s averaged power")
    ap.add_argument("--poll-hz", type=int, default=100)
    ap.add_argument("--dram-buf-bytes", type=int, default=None,
                    help="default: max(1 GiB, 64*L2)")
    ap.add_argument("--l2-buf-bytes", type=int, default=None,
                    help="default: max(4 MiB, 25%% of L2)")
    ap.add_argument("--l2-fraction", type=float, default=0.25)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--blocks-per-sm", type=int, default=32)
    ap.add_argument("--cal-passes", type=int, default=4)
    ap.add_argument("--cal-repeats", type=int, default=3)
    ap.add_argument("--l2-warm-passes", type=int, default=4,
                    help="L2 buffer warmup passes immediately before each L2 phase")
    ap.add_argument("--l2-cache-op", choices=["ca", "cg", "cs"], default="cg")
    ap.add_argument("--dram-cache-op", choices=["ca", "cg", "cs"], default="cs")
    ap.add_argument("--include-compute", action="store_true",
                    help="also run a no-global-memory FMA reference stage")
    ap.add_argument("--fit-aggregate", choices=["median", "raw"], default="median",
                    help="median aggregates repeated target points before slope fitting")
    ap.add_argument("--fit-min-target", type=int, default=50)
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle stage/target order within each repeat to reduce drift bias")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--hbm-pjbit", type=float, default=None,
                    help="external HBM stack+PHY prior to subtract; NVML does not measure this separately")
    ap.add_argument("--only-stage", choices=["control_l2", "l2", "control_dram", "dram", "compute"], default="",
                    help="run one stage only; mainly useful for Nsight Compute validation")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--flat-output", action="store_true")
    ap.add_argument("--tag", default="read_decomp")
    ap.add_argument("--r2-warn", type=float, default=0.98)
    ap.add_argument("--clock-warn-mhz", type=float, default=75.0)
    ap.add_argument("--temperature-warn-c", type=float, default=80.0,
                    help="warn when active phase mean GPU temperature exceeds this value")
    ap.add_argument("--idle-iqr-warn-w", type=float, default=2.0)
    ap.add_argument("--min-dram-l2-multiple", type=int, default=64)
    ap.add_argument("--set-persistence", action="store_true",
                    help="run nvidia-smi -pm 1 before measurement; requires root")
    ap.add_argument("--lock-gpu-clocks", default="",
                    help="value for nvidia-smi -lgc, e.g. 1410,1410; requires root")
    ap.add_argument("--lock-memory-clocks", default="",
                    help="value for nvidia-smi -lmc; Hopper may require --lock-memory-clocks-deferred")
    ap.add_argument("--lock-memory-clocks-deferred", default="",
                    help="value for nvidia-smi -lmcd; takes effect at next GPU initialization")
    ap.add_argument("--reset-gpu-clocks-before", action="store_true")
    ap.add_argument("--reset-memory-clocks-before", action="store_true")
    ap.add_argument("--reset-clocks-on-exit", action="store_true")
    ap.add_argument("--power-limit-w", type=float, default=None)
    ap.add_argument("--power-limit-scope", type=int, choices=[0, 1], default=None)
    args = ap.parse_args()

    if args.poll_hz <= 0:
        raise SystemExit("--poll-hz must be positive")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.warmup_repeats < 0:
        raise SystemExit("--warmup-repeats must be non-negative")
    if args.repeats <= args.warmup_repeats:
        print("[warn] --repeats <= --warmup-repeats; no fitted points will remain")
    if args.l2_fraction <= 0 or args.l2_fraction >= 1.0:
        raise SystemExit("--l2-fraction must be in (0,1)")
    bad = [t for t in args.targets if t < 0 or t > 100]
    if bad:
        raise SystemExit(f"targets must be in 0..100: {bad}")

    args.targets = list(dict.fromkeys(args.targets))
    env_actions = configure_gpu_with_nvidia_smi(args)
    for action in env_actions:
        status = "ok" if int(action.get("returncode", -1)) == 0 else "warn"
        print(f"[{status}] {action.get('label')}: {action.get('cmd', '')}")
        if int(action.get("returncode", -1)) != 0 and action.get("stderr"):
            print(f"[warn] {action.get('stderr')}")

    cp.cuda.Device(args.device).use()
    props = cp.cuda.runtime.getDeviceProperties(args.device)
    gpu_name = base.prop(props, "name")
    sm_count = int(props["multiProcessorCount"])
    l2_size_bytes = int(props["l2CacheSize"])
    args.l2_size_bytes = l2_size_bytes

    if args.dram_buf_bytes is None:
        args.dram_buf_bytes = max(1 << 30, l2_size_bytes * 64)
    args.dram_buf_bytes = align16(args.dram_buf_bytes)
    if args.l2_buf_bytes is None:
        args.l2_buf_bytes = max(4 << 20, int(l2_size_bytes * args.l2_fraction))
    args.l2_buf_bytes = min(align16(args.l2_buf_bytes), args.dram_buf_bytes)

    blocks = args.blocks if args.blocks is not None else sm_count * args.blocks_per_sm
    if args.threads % 32 != 0:
        raise SystemExit("--threads must be a multiple of 32")
    sink_u4 = max(1024, blocks * (args.threads // 32))

    print(f"[info] GPU={gpu_name} SMs={sm_count} L2={l2_size_bytes/(1<<20):.1f} MiB")
    print(f"[info] buffers: l2={args.l2_buf_bytes/(1<<20):.2f} MiB "
          f"dram={args.dram_buf_bytes/(1<<30):.2f} GiB")
    print(f"[info] blocks={blocks} threads={args.threads} targets={args.targets} repeats={args.repeats}")
    if base._nvrtc_path:
        print(f"[info] preloaded nvrtc: {base._nvrtc_path}")

    dram_buf = cp.empty(args.dram_buf_bytes // 4, dtype=cp.uint32)
    l2_buf = cp.empty(args.l2_buf_bytes // 4, dtype=cp.uint32)
    sink = cp.empty(sink_u4 * 4, dtype=cp.uint32)
    dram_buf.fill(np.uint32(0x3c6ef372))
    l2_buf.fill(np.uint32(0x9e3779b9))
    sink.fill(np.uint32(0))
    buffers = {"dram": dram_buf, "l2": l2_buf}

    module = cp.RawModule(code=KERNEL_CODE, options=("--std=c++14",))
    kernels = {
        "decomp_control_read": module.get_function("decomp_control_read"),
        "decomp_stream_read_ca": module.get_function("decomp_stream_read_ca"),
        "decomp_stream_read_cg": module.get_function("decomp_stream_read_cg"),
        "decomp_stream_read_cs": module.get_function("decomp_stream_read_cs"),
        "decomp_compute_fma": module.get_function("decomp_compute_fma"),
    }
    stream = cp.cuda.Stream(non_blocking=True)
    if args.only_stage == "compute" and not args.include_compute:
        args.include_compute = True
        print("[info] --only-stage=compute requested; enabling --include-compute automatically")
    specs = make_specs(args.l2_buf_bytes, args.dram_buf_bytes,
                       args.l2_cache_op, args.dram_cache_op, args.include_compute)
    if args.only_stage:
        specs = [s for s in specs if s.stage == args.only_stage]
        if not specs:
            all_specs = make_specs(args.l2_buf_bytes, args.dram_buf_bytes,
                                   args.l2_cache_op, args.dram_cache_op, True)
            valid = sorted({s.stage for s in all_specs})
            raise SystemExit(
                f"--only-stage={args.only_stage!r} matched no specs. "
                f"Valid stages: {valid}. Did you forget --include-compute?"
            )
        print(f"[info] only-stage={args.only_stage}")
        print("[warn] --only-stage is a validation/NCU mode; complete decomposition requires control_l2,l2,control_dram,dram")
    if not specs:
        raise SystemExit("no workloads selected")

    # Warmup all kernels and populate L2 outside measurement.
    for spec in specs:
        launch(spec, kernels, stream, blocks, args.threads, buffers, sink, 1)
    stream.synchronize()

    calibration: dict[str, dict[str, float | int | str]] = {}
    for spec in specs:
        ms, pps = calibrate(spec, kernels, stream, blocks, args.threads, buffers, sink,
                            args.cal_passes, args.cal_repeats, args.l2_warm_passes)
        calibration[spec.stage] = {
            "name": spec.name,
            "ms_per_pass": ms,
            "passes_per_second": pps,
            "nominal_peak_bandwidth_gbs": spec.buf_bytes * pps / 1e9,
            "buffer_bytes": spec.buf_bytes,
            "n_uint4": spec.n_u4,
        }
        print(f"[calib] {spec.stage:<8} {ms:.3f} ms/pass  "
              f"~{calibration[spec.stage]['nominal_peak_bandwidth_gbs']:.1f} GB/s nominal")

    cp.cuda.runtime.deviceSynchronize()
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.device)
    metadata_before = base.nvml_snapshot(handle)
    if args.idle_settle_seconds > 0:
        print(f"[idle] settling for {args.idle_settle_seconds:.1f} s before baseline ...", flush=True)
        time.sleep(args.idle_settle_seconds)
    print(f"[idle] measuring pre-idle baseline for {args.idle_seconds:.1f} s ...", flush=True)
    idle_stats = measure_idle_power_stats(
        handle, args.idle_seconds, args.poll_hz,
        args.idle_filter_sm_clock_below, args.idle_filter_min_samples)
    idle_power_w = float(idle_stats["mean_w"])
    print(f"[idle] mean={idle_stats['mean_w']:.2f} W median={idle_stats['median_w']:.2f} W "
          f"IQR={idle_stats['iqr_w']:.2f} W std={idle_stats['std_w']:.2f} W "
          f"n={idle_stats['samples']} source={idle_stats['baseline_source']} "
          f"sm_clock_range={idle_stats['sm_clock_mhz_min']}..{idle_stats['sm_clock_mhz_max']} MHz "
          f"pstates={idle_stats['pstates_seen']}", flush=True)

    rng = random.Random(args.seed)
    rows: list[PhaseRow] = []
    metadata_after: dict = {}
    poller = base.PowerPoller(handle, args.poll_hz)
    poller.start()
    try:
        for rep in range(args.repeats):
            seq = [(spec, target) for target in args.targets for spec in specs]
            if args.shuffle:
                rng.shuffle(seq)
            for spec, target in seq:
                print(f"[phase] repeat={rep} stage={spec.stage} target={target}% start")
                row = run_phase(
                    spec, rep, target, kernels, stream, blocks, args.threads,
                    buffers, sink, args.phase_seconds, args.window_ms,
                    float(calibration[spec.stage]["ms_per_pass"]),
                    poller, idle_power_w, args.l2_warm_passes)
                rows.append(row)
                print(f"[phase] {row.phase:<24} BW={row.nominal_bandwidth_gbs:.1f} GB/s "
                      f"Pavg={row.avg_power_w:.1f} W Pdyn={row.dynamic_power_w:.1f} W "
                      f"pJ/nominal-bit={base.fmt_or_na(row.pj_per_nominal_bit)}")
                with nvtx.annotate("gap", color="gray"):
                    poller.set_phase("gap")
                    time.sleep(max(0.0, args.gap_seconds))
    finally:
        poller.stop()
        metadata_after = base.nvml_snapshot(handle)
        pynvml.nvmlShutdown()
        cleanup_actions = cleanup_gpu_with_nvidia_smi(args)
        env_actions.extend(cleanup_actions)

    run_stamp_minute = time.strftime("%Y%m%d%H%M")
    out_dir = base.resolve_output_dir(args.out_dir, gpu_name, args.flat_output, run_stamp_minute)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    stem = f"read_energy_decomp_{base.safe_name(gpu_name)}_{stamp}{suffix}"

    stages_for_fit: list[str] = []
    for spec in specs:
        if spec.stage not in stages_for_fit:
            stages_for_fit.append(spec.stage)
    aggregate_rows = make_aggregate_rows(rows, args.warmup_repeats)
    fits = [f for f in (fit_power_vs_bw(stage, rows, args.fit_min_target,
                                        args.warmup_repeats, args.fit_aggregate)
                       for stage in stages_for_fit) if f is not None]
    fit_by_stage = {f.stage: f for f in fits}
    decomp = build_decomp(fit_by_stage, args.hbm_pjbit, args.l2_cache_op, args.dram_cache_op)
    quality = quality_checks(rows, fits, args, idle_stats, env_actions)

    summary_csv = out_dir / f"{stem}_summary.csv"
    trace_csv = out_dir / f"{stem}_trace.csv"
    aggregate_csv = out_dir / f"{stem}_aggregate.csv"
    fits_csv = out_dir / f"{stem}_fits.csv"
    decomp_csv = out_dir / f"{stem}_decomposition.csv"
    quality_csv = out_dir / f"{stem}_quality.csv"
    metadata_json = out_dir / f"{stem}_metadata.json"
    power_png = out_dir / f"{stem}_power.png"
    fit_png = out_dir / f"{stem}_fit.png"
    diagnostics_png = out_dir / f"{stem}_diagnostics.png"
    method_png = out_dir / f"{stem}_method.png"
    decomp_png = out_dir / f"{stem}_decomposition.png"

    save_rows(summary_csv, rows)
    save_trace(trace_csv, poller.samples)
    save_rows(aggregate_csv, aggregate_rows)
    save_rows(fits_csv, fits)
    save_rows(decomp_csv, decomp)
    save_rows(quality_csv, quality)
    save_power_plot(power_png, gpu_name, rows, poller.samples, idle_power_w)
    save_fit_plot(fit_png, gpu_name, rows, aggregate_rows, fits,
                  args.warmup_repeats, args.fit_min_target, args.fit_aggregate)
    save_diagnostics_plot(diagnostics_png, gpu_name, rows, args)
    save_method_plot(method_png, gpu_name, decomp, args.l2_cache_op, args.dram_cache_op, args.hbm_pjbit)
    save_decomp_plot(decomp_png, gpu_name, fits, decomp)

    metadata = {
        "args": vars(args),
        "device": {
            "name": gpu_name,
            "sm_count": sm_count,
            "l2_bytes": l2_size_bytes,
            "blocks": blocks,
            "threads": args.threads,
            "sink_uint4": sink_u4,
        },
        "calibration": calibration,
        "idle": {
            "power_w_mean": idle_power_w,
            "power_w_std": idle_stats["std_w"],
            "power_w_median": idle_stats["median_w"],
            "power_w_iqr": idle_stats["iqr_w"],
            "samples": idle_stats["samples"],
            "baseline_source": idle_stats.get("baseline_source", ""),
            "all_samples": idle_stats.get("all_samples", idle_stats.get("samples", 0)),
            "filtered_samples": idle_stats.get("filtered_samples", 0),
            "sm_clock_filter_mhz": idle_stats.get("sm_clock_filter_mhz", 0),
            "sm_clock_mhz_min": idle_stats.get("sm_clock_mhz_min", -1),
            "sm_clock_mhz_max": idle_stats.get("sm_clock_mhz_max", -1),
            "pstates_seen": idle_stats.get("pstates_seen", ""),
        },
        "nvml_before": metadata_before,
        "nvml_after": metadata_after,
        "env_actions": env_actions,
        "plots": {
            "power_trace": str(power_png),
            "fit_regression": str(fit_png),
            "diagnostics": str(diagnostics_png),
            "method_diagram": str(method_png),
            "decomposition": str(decomp_png),
        },
        "method_notes": [
            "Recommended values come from avg_power vs bandwidth slopes over active phases, not P8 idle subtraction.",
            "Idle baseline can prefer low-SM-clock samples for diagnostic dynamic_power columns; fits do not depend on idle subtraction.",
            "--only-stage is intended for Nsight Compute validation and produces partial stage-slope output, not a complete decomposition.",
            "Temperature and clock quality warnings should be checked before comparing A100/H100 slopes.",
            "Legacy control_* stage names mean matched no-read baseline loops, not GPU control units.",
            "baseline/l2/dram stages use the same grid, loop shape, and accumulator sink convention.",
            "l2 phases are warmed immediately before measurement to improve L2 residency.",
            "dram phases rely on DRAM buffer much larger than L2 and a streaming cache-op hint to make L1/L2 hits unlikely.",
            "ld.global.ca caches at all available cache levels and can include L1/TEX effects; validate with Nsight Compute l1tex/lts counters.",
            "ld.global.cs is an evict-first/streaming hint, not an architectural guarantee of L2 bypass.",
            "Visualization plots: _fit.png is the primary slope-readout plot; _diagnostics.png exposes clock/thermal drift; _method.png explains component formulas.",
            "compute stage, when enabled, is a no-global-memory SM/FMA dynamic reference and is not subtracted from HBM.",
            "Use Nsight Compute physical dram/lts bytes to validate physical denominators and cache residency.",
            "--hbm-pjbit is an external HBM prior; NVML does not separately measure HBM stack energy.",
        ],
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str))

    print(f"[save] {summary_csv}")
    print(f"[save] {aggregate_csv}")
    print(f"[save] {fits_csv}")
    print(f"[save] {decomp_csv}")
    print(f"[save] {quality_csv}")
    print(f"[save] {trace_csv}")
    print(f"[save] {metadata_json}")
    print(f"[save] {power_png}")
    print(f"[save] {fit_png}")
    print(f"[save] {diagnostics_png}")
    print(f"[save] {method_png}")
    print(f"[save] {decomp_png}")
    print_table(fits, decomp, quality)


if __name__ == "__main__":
    main()
