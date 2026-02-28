# wgpu prover

GPU-accelerated [Groth16](https://eprint.iacr.org/2016/260) zero-knowledge proof system
built on [wgpu](https://github.com/gfx-rs/wgpu), targeting the BLS12-381 curve.

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
(2) workgroup tree reduction to sum each window.

**NTT pipeline** — tile-based Cooley-Tukey FFT with pre-computed twiddle factors in
Montgomery form, bit-reversal permutation, and coset shift stages.

### Features

- **Cross-platform GPU** — runs on Metal (macOS), Vulkan (Linux), DX12 (Windows), and
  WebGPU (WASM) via wgpu.
- **BLS12-381 native** — 384-bit base field and 256-bit scalar field arithmetic implemented
  entirely in WGSL compute shaders.
- **`PreparedProvingKey`** — pre-serializes proving key base points to GPU format once,
  amortising the O(N) conversion cost across multiple proofs.
- **Pluggable bellman backend** — compile with `bellman-provider-bellman` or the default
  `bellman-provider-nam-bellperson`.

## Optimizations

The following optimizations have been applied to the GPU proving pipeline, listed in
chronological order. Benchmarks measured on an Apple M3 Max.

### 1. Gap-skipping subsum accumulation (G1)
The `subsum_accumulation_g1` shader was iterating O(2^c) = 32,768 times per window
regardless of input size. Replaced with gap-skipping double-and-add: only active buckets
are visited and empty runs are skipped in O(log(gap)) via scalar multiplication.
- **full_proof:** 22.4s → 131ms (170x)
- **msm_g1/n=100:** 23.6s → 1.39s (17x)

### 2. Smaller bucket width for G2 MSM
G2 extension field operations cause register pressure in Metal shaders, preventing
gap-skipping for G2. Instead, reduced bucket width from c=15 to c=4:
- c=15: 18 windows × 32,768 iterations = 589K EC additions per MSM
- c=4: 64 windows × 15 iterations = 960 EC additions (614x fewer)
- **msm_batch/5x100:** 77.9s → 2.92s (27x)

### 3. Parallel bucket weighting
Moved scalar multiplication weighting (`v * B[v]`) from the sequential subsum pass
(18 threads) into the parallel aggregate pass (~1,800 threads). Each bucket now
computes its weighted contribution during aggregation.
- **msm_g1/n=100:** 1.38s → 113ms (12x)

### 4. Montgomery form pre-conversion
Added a compute pre-pass that converts G1 base points to Montgomery form in-place once
per MSM, so `aggregate_buckets_g1` can skip the 3 Montgomery multiplications per point
load. Saves 3·N·(W−1) field multiplications.

### 5. Mixed affine + projective addition
Added `add_g1_mixed()` that exploits Z₂ = R (affine points in Montgomery form) to save
5 Montgomery multiplications per point addition (11 vs 16 muls). Used in
`aggregate_buckets_g1` where base points are always affine (~31% fewer field muls per
addition).

### 6. PreparedProvingKey
Introduced `PreparedProvingKey` that pre-serializes all proving key base points (a, b_g1,
l, h, b_g2) to GPU byte format once. Subsequent proofs skip the per-proof O(N) CPU
serialization loop.

### 7. Parallel subsum via workgroup tree reduction
Replaced the single-threaded `subsum_accumulation_g1` (workgroup_size(1)) with a parallel
tree reduction using 64 threads per window. Each thread sums its strided subset of
pre-weighted buckets, then a 6-stage binary reduction in shared memory produces the final
window sum.
- **msm_g1/n=10K:** 6.28s → 440ms (14.3x)
- **full_proof/n=10K:** 25s → 4.9s (5.1x)

### 8. Pre-computed scalar windows
Previously `scalar_to_windows()` was called N × num_windows times (once per scalar per
window iteration). Now windows are pre-computed once into a flat Vec, reducing allocations
from 180K to 10K at n=10K.
- **bucket_sort:** 170ms → 18.5ms (9.2x)
- **msm_g1/n=10K:** 440ms → 287ms

### 9. Early-exit scalar multiplication
The `scalar_mul_g1` shader always iterated 16 times (one per bit of the bucket width)
even when the scalar was small. Added fast paths for k=0 (return identity) and k=1
(return point unchanged), plus an early break when the remaining scalar bits are zero.
For uniformly distributed bucket values in [1, 32767], this saves ~1-2 loop iterations
on average, with bigger wins for the many k=1 buckets.

### 10. CPU-GPU overlap for H polynomial
Split `compute_h_poly` into a non-blocking GPU submit phase and a separate read phase.
While the GPU processes the H polynomial pipeline (~167ms), the CPU pre-computes bucket
sorting data for the 4 non-H MSMs (a, b1, l, b2) in parallel. The H bucket data is
computed after the GPU result is read back. This hides ~74ms of CPU bucket sorting
behind the GPU wait.

## Discarded optimizations

The following optimizations were investigated but ultimately discarded because they
showed no improvement or caused regressions.

### GPU window folding
**Idea:** Fold the per-window sums on the GPU instead of reading them back to the CPU
and folding there.

**Why discarded:** Window folding is inherently sequential — each window's contribution
depends on the previous result after c-bit shifts. The CPU cost is negligible (~1ms
for 18 windows), so the GPU dispatch overhead would exceed any savings.

### Batch MSM submissions
**Idea:** Combine all 5 MSM `queue.submit()` calls into a single submission to reduce
GPU scheduling overhead.

**Why discarded:** Caused a regression. The current separate submissions create a natural
CPU-GPU pipeline: while the GPU executes one MSM, the CPU prepares buffers and bucket
data for the next. Batching all submissions forces the CPU to finish all preparation
before any GPU work begins, eliminating this overlap.

### Reduce field inversions
**Idea:** Batch field inversions in EC point addition using Montgomery's trick to
amortize inversions across multiple additions.

**Why discarded:** Not applicable. All EC point additions use projective coordinates,
which avoid field inversions entirely. The only inversion occurs in the final
projective-to-affine conversion.

### Buffer pooling
**Idea:** Reuse GPU buffers across MSMs instead of creating new ones each time.

**Why discarded:** wgpu buffer creation is already very fast (sub-microsecond). The
complexity of managing a buffer pool would not pay off, and buffer sizes vary per MSM
making reuse impractical.

### Sort buckets by descending size
**Idea:** Sort active buckets by size (largest first) within each window before
dispatching to the GPU. This would group similarly-sized buckets into the same warps,
reducing warp divergence where fast threads idle while the slowest thread finishes.

**Why discarded:** Showed a +1.7% regression at n=10K. At this scale, ~2500 active
buckets have an average size of only 1-2 points — there is no meaningful variance
to sort. The sorting overhead (Vec allocation + sort) outweighed any divergence
reduction.

### G2 tree reduction (parallel subsum)
**Idea:** Apply the same 64-thread parallel tree reduction used for G1 subsum to G2.
This requires pre-weighting buckets with `scalar_mul_g2` (v * B[v]) in the aggregate
pass, then summing the weighted buckets in parallel. G2 subsum currently uses a
single-threaded Pippenger running-sum with `@workgroup_size(1)`.

**Why discarded:** The Metal shader compiler non-deterministically miscompiles
`double_g2` when called from `add_g2`'s equal-point detection path. Any form of
scalar multiplication on G2 points (k * P for k ≥ 2) triggers `P + P`, which calls
`double_g2` through `add_g2`. Multiple workarounds were attempted: repeated addition,
separate weighting kernel, and reduced-register-pressure `double_g2` rewrite — all
produced intermittent invalid G2 point outputs on Metal. The original Pippenger
running-sum avoids this by never adding equal points.
