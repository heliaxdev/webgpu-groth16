# webgpu-groth16

GPU-accelerated [Groth16](https://eprint.iacr.org/2016/260) zero-knowledge proof system
built on [wgpu](https://github.com/gfx-rs/wgpu), with curve-pluggable abstractions.
The current production implementation targets BLS12-381.

## Overview

The two most expensive operations in Groth16 proving are multi-scalar multiplication (MSM)
and the Number Theoretic Transform (NTT). This library offloads both to GPU compute shaders
written in WGSL, while the rest of the proof construction (constraint synthesis, witness
generation, random blinding) stays on the CPU.

### Architecture

```
Circuit (bellman ConstraintSystem)
│
▼
Witness & R1CS coefficients  ──►  CPU
│
├── MSM  ──────────────────►  GPU (2-stage: bucket aggregation → parallel tree reduction)
├── NTT / iNTT  ──────────►  GPU (Cooley-Tukey FFT with coset shifts)
└── H(x) polynomial  ─────►  GPU (pointwise A·B−C, division by Z_H)
│
▼
Groth16 proof (A, B, C)

```

**MSM pipeline** — scalars are decomposed into windows on the CPU (bucket sorting), then two
GPU passes run: (1) parallel bucket aggregation using mixed affine+projective addition, and
(2) workgroup tree reduction to sum each window. For curves that expose GLV support,
G1 decomposition uses the curve-provided endomorphism path.

**NTT pipeline** — tile-based Cooley-Tukey FFT with pre-computed twiddle factors in
Montgomery form, bit-reversal permutation, and coset shift stages.

### Features

- **Cross-platform GPU** — runs on Metal (macOS), Vulkan (Linux), DX12 (Windows), and
  WebGPU (WASM) via wgpu.
- **Curve-pluggable design** — generic `GpuCurve` trait for curve-specific arithmetic,
  serialization layout, and GPU dispatch parameters; BLS12-381 is fully supported today.
- **`PreparedProvingKey`** — pre-serializes proving key base points to GPU format once,
  amortising the O(N) conversion cost across multiple proofs.
- **Pluggable bellman backend** — compile with `bellman-provider-bellman` or the default
  `bellman-provider-nam-bellperson`.

## Profiling

GPU-level profiling is available via [wgpu-profiler](https://github.com/Wumpf/wgpu-profiler),
gated behind the `profiling` feature. Results are written as a Chrome trace JSON file.

```bash
# Run the profiling harness (default: 10K constraints, 5 iterations)
cargo run --release --example profile --features profiling -- [NUM_SQUARINGS] [ITERATIONS]

# Open the generated trace in [https://ui.perfetto.dev](https://ui.perfetto.dev) → load profile.json

```

Instrumented GPU passes:

* **H pipeline** — `to_montgomery`, `intt_abc`, `coset_shift_abc`, `ntt_abc`,
`pointwise_poly`, `intt_h`, `inv_coset_shift_h`, `from_montgomery_h`
* **MSM** (G1/G2) — `to_montgomery_bases`, `bucket_aggregation`, `tree_reduction`

## Benchmarks

Latest [Criterion](https://bheisler.github.io/criterion.rs/book/) results on an
Apple M3 Max (`cargo bench`).

### Full proof (Groth16 end-to-end)

Repeated-squaring circuit with `n` constraints — includes witness generation,
NTT/H-polynomial, all 5 MSMs (a, b_g1, l, h, b_g2), and proof assembly.

| Constraints | Mean |
| --- | --- |
| 2 | 85 ms |
| 1K | 266 ms |
| 10K | 690 ms |
| 100K | 2.47 s |

### MSM batch (5 MSMs: a, b_g1, l, h, b_g2)

Isolated MSM batch — same 5-MSM workload as a real proof, with random bases and scalars.

| Points per MSM | Mean |
| --- | --- |
| 100 | 165 ms |
| 1K | 259 ms |
| 10K | 642 ms |
| 100K | 2.10 s |

### Bucket sorting (CPU)

Signed-digit scalar decomposition and bucket assignment (CPU-only, no GPU).

| Points | Mean |
| --- | --- |
| 1K | 2.38 ms |
| 10K | 17.8 ms |
| 100K | 139 ms |

### MASP Sapling Output circuit

Real-world proof using the MASP Sapling Output circuit (31,211 constraints,
n=32,768).

| Phase | Time |
| --- | --- |
| Synthesis | 14.5 ms |
| H polynomial (GPU) | 152 ms |
| Bucket sorting (CPU, GLV) | 92.1 ms |
| MSM batch (GPU) | 781 ms |
| **Total proof** | **1.04 s** |

Proving key sizes: a=28,759, b_g1=21,384, l=30,896, h=32,767, b_g2=21,384.

Run all benchmarks:

```bash
cargo bench

```

## Latest Benchmark Results

Measured on Apple M3 Max. Criterion median times.

| Benchmark | Time |
| --- | --- |
| proof/n=2 | 80.1 ms |
| proof/n=1000 | 229 ms |
| proof/n=10000 | 601 ms |
| proof/n=100000 | 2.32 s |
| msm_g1/n=100 | 22.6 ms |
| msm_g1/n=1000 | 33.0 ms |
| msm_g1/n=10000 | 98.1 ms |
| msm_g1/n=100000 | 339 ms |
| msm_batch/5x100 | 144 ms |
| msm_batch/5x1000 | 232 ms |
| msm_batch/5x10000 | 561 ms |
| msm_batch/5x100000 | 2.04 s |
| bucket_sorting/n=1000 | 2.31 ms |
| bucket_sorting/n=10000 | 17.5 ms |
| bucket_sorting/n=100000 | 139 ms |
| ntt/h_poly_n=8 | 4.07 ms |
| ntt/h_poly_n=1024 | 11.4 ms |
| ntt/h_poly_n=16384 | 113 ms |
| sapling_output/proof | 987 ms |
