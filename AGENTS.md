# Agent Instructions

## Project Overview

This is a GPU-accelerated Groth16 zero-knowledge proof system built on wgpu, with a generic `GpuCurve` abstraction and a production BLS12-381 implementation. The two most expensive operations — multi-scalar multiplication (MSM) and the Number Theoretic Transform (NTT) — are offloaded to GPU compute shaders written in WGSL. The rest of the proof construction (constraint synthesis, witness generation, random blinding) runs on the CPU. Make sure to only allow optimizations that are compatible with wasm.

Key implementation details (current BLS12-381 backend):
- Field arithmetic uses 13-bit limbs for F_q (30x13-bit) and 32-bit limbs for F_r (8x32-bit)
- MSM uses Pippenger bucket method with signed-digit scalar decomposition (c=13 for G1, c=8 for G2)
- G1 MSM uses multi-workgroup tree reduction; G2 MSM uses single-threaded running-sum due to a Metal shader compiler bug with `double_g2`
- WGSL shaders are concatenated via `concat!(include_str!(...))` using split curve/MSM modules:
  - G1 MSM: `fr.wgsl + fp.wgsl + curve_g1.wgsl + msm_g1_*.wgsl`
  - G2 MSM: `fr.wgsl + fp.wgsl + curve_g2.wgsl + msm_g2_*.wgsl`

## Testing Requirements

All changes must include tests. Run the existing test suite before and after your changes to ensure nothing is broken:

```bash
cargo test --release
```

Use `--release` mode — debug builds are prohibitively slow for GPU workloads.

## Benchmarking Requirements

All changes must be benchmarked. Run benchmarks before and after your changes to measure the impact:

```bash
cargo bench
```

If a new feature or optimization is added, the benchmark results must be documented in the **Optimizations** section of `README.md`, following the existing format. Each entry should include:
- A descriptive title
- An explanation of what was changed and why
- Before/after benchmark numbers with the speedup factor

If an optimization is attempted but shows no improvement or causes regressions, document it in the **Discarded optimizations** section of `README.md` instead, explaining the idea and why it was discarded. Always try to optimize the `bench_gpu_sapling_output` real world benchmark, this is what we want need to make faster.

After any change (feature, optimization, or bug fix), update or add a **Latest Benchmark Results** section in `README.md` with the current benchmark numbers so the README always reflects the up-to-date performance of the prover.
