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

## Profiling

GPU-level profiling is available via [wgpu-profiler](https://github.com/Wumpf/wgpu-profiler)
and [Tracy](https://github.com/wolfpld/tracy), gated behind the `profiling` feature.

```bash
# Run the profiling harness (default: 10K constraints, 5 iterations)
cargo run --release --example profile --features profiling -- [NUM_SQUARINGS] [ITERATIONS]

# Connect Tracy profiler to view the flamegraph
```

Tracy shows a live flamegraph with per-compute-pass GPU timing (H pipeline stages,
MSM bucket aggregation / tree reduction).

Instrumented GPU passes:
- **H pipeline** — `to_montgomery`, `intt_abc`, `coset_shift_abc`, `ntt_abc`,
  `pointwise_poly`, `intt_h`, `inv_coset_shift_h`, `from_montgomery_h`
- **MSM** (G1/G2) — `to_montgomery_bases`, `bucket_aggregation`, `tree_reduction`

## Benchmarks

Latest [Criterion](https://bheisler.github.io/criterion.rs/book/) results on an
Apple M3 Max (`cargo bench`).

### Full proof (Groth16 end-to-end)

Repeated-squaring circuit with `n` constraints — includes witness generation,
NTT/H-polynomial, all 5 MSMs (a, b_g1, l, h, b_g2), and proof assembly.

| Constraints | Mean    |
|-------------|---------|
| 2           | 140 ms  |
| 1K          | 1.47 s  |
| 10K         | 2.18 s  |
| 100K        | 6.00 s  |

### MSM batch (5 MSMs: a, b_g1, l, h, b_g2)

Isolated MSM batch — same 5-MSM workload as a real proof, with random bases and scalars.

| Points per MSM | Mean    |
|-----------------|---------|
| 100             | 1.35 s  |
| 1K              | 1.44 s  |
| 10K             | 2.16 s  |
| 100K            | 5.26 s  |

### Bucket sorting (CPU)

Signed-digit scalar decomposition and bucket assignment (CPU-only, no GPU).

| Points | Mean    |
|--------|---------|
| 1K     | 14.6 ms |
| 10K    | 18.5 ms |
| 100K   | 160 ms  |

### MASP Sapling Output circuit

Real-world proof using the MASP Sapling Output circuit (31,211 constraints,
n=32,768).

| Phase                  | Time      |
|------------------------|-----------|
| Synthesis              | 14.5 ms   |
| H polynomial (GPU)     | 151 ms    |
| Bucket sorting (CPU)   | 97.6 ms   |
| MSM batch (GPU)        | 16.7 s    |
| **Total proof**        | **16.9 s** |

Proving key sizes: a=28,759, b_g1=21,384, l=30,896, h=32,767, b_g2=21,384.

Run all benchmarks:

```bash
cargo bench
```

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

### 11. Mixed affine + projective G2 addition
Added `add_g2_mixed()` that exploits Z₂ = (R,0) (affine points in Fq2 Montgomery form)
to save 5 Fq2 multiplications per point addition (11 vs 16 mul_fp2). Combined with a
`to_montgomery_bases_g2` pre-pass that converts G2 base points to Montgomery form
once per MSM, eliminating 6 redundant Montgomery conversions per point per window.
- **msm_batch/5x100K:** 37.7s → 25.3s (1.49x)
- **full_proof/n=100K:** 38.1s → 25.9s (1.47x)

### 12. Larger G2 bucket width (c=4 → c=8)
Increased G2 MSM bucket width from c=4 to c=8, reducing the number of windows from
64 to 32 and improving GPU parallelism. With c=4, only 960 active buckets were
dispatched (15 per window × 64 windows), each processing ~6,667 points sequentially.
With c=8, 8,160 active buckets (255 per window × 32 windows) process ~392 points
each, achieving much better GPU utilization. Profiled c=4,6,8,10 — c=8 was the
clear winner; c=10 regressed due to O(2^c) single-threaded subsum cost.
- **msm_batch/5x100K:** 25.3s → 6.98s (3.6x)
- **full_proof/n=100K:** 25.9s → 7.7s (3.4x)

### 13. Multi-workgroup G1 tree reduction
GPU profiling revealed that G1 tree_reduction consumed ~49% of MSM time — nearly
equal to bucket_aggregation. The single-workgroup subsum dispatched only 18 workgroups
× 64 threads = 1,152 threads on an M3 Max with ~4,000 shader cores (29% utilization).
Split into a two-pass approach: Phase 1 dispatches `num_windows × 32` workgroups,
each reducing a chunk of pre-weighted buckets into partial sums; Phase 2 reduces 32
partial sums per window into the final window sum.
- **proof/n=10K:** 2.55s → 2.26s (-12.6%)
- **proof/n=100K:** 7.72s → 6.42s (-16.8%)
- **msm_batch/5x100K:** 6.98s → 6.17s (-11.6%)

### 14. Optimal G1 bucket width (c=15 → c=13)
Profiled G1 bucket width at c=11..16. Results showed a bimodal pattern: c=11,12,14
caused catastrophic regressions (30-54s), while c=13 and c=15 performed well. c=13
was the fastest, likely due to better balance between bucket count, GPU occupancy,
and per-thread work on the Metal GPU scheduler.
- **proof/n=10K:** 2.26s → 2.18s (-3.5%)
- **proof/n=100K:** 6.42s → 5.97s (-7.0%)
- **msm_batch/5x100K:** 6.17s → 5.26s (-14.7%)

### 15. 13-bit limbs with scalar accumulator Montgomery multiplication
Switched Fq (384-bit base field) from 12×32-bit limbs (R=2^384) to 30×13-bit limbs
(R=2^390). This eliminates the `mul_u32` 16-bit decomposition hack — native u32×u32
products fit in 26 bits, so each limb×limb multiply is a single instruction.

The critical change was using 32 scalar accumulator variables (`var t0: u32 = 0u; ...
var t31: u32 = 0u;`) instead of `var t = array<u32, 32>(...)`. With an array, the Metal
GPU compiler spills the entire accumulator to device memory; with individual scalars, it
can freely allocate, reorder, and selectively spill individual values.

The optimized `mul_montgomery_u384` and `sqr_montgomery_u384` functions are fully
unrolled (multiply + reduce phases, carry propagation) with literal `Q_MODULUS`
constants (no array indexing). A dedicated `sqr_montgomery_u384` takes a single input,
saving 30 registers vs `mul(a, a)`. Both are generated by `scripts/gen_fp_unrolled.py`.
- **msm_batch/5x100K:** 25.5s → 16.9s (1.5x)

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

### GLV endomorphism for G1 MSM
**Idea:** Use the BLS12-381 GLV endomorphism φ(x,y) = (β·x, y) to decompose each
256-bit scalar k into two ~128-bit halves k1, k2 such that k·P = k1·P + k2·φ(P).
This halves the Pippenger window count from 18 to 9 (ceil(128/15) vs ceil(256/15)),
reducing subsum accumulation dispatches by 50% and CPU fold doublings from 255 to 120.

**Why discarded:** Bucket aggregation work is unchanged — 18 windows × N points =
9 windows × 2N points. Since bucket aggregation dominates the MSM cost (~97% of
msm_batch time), the subsum and fold savings are negligible. The per-proof CPU
overhead of GLV decomposition + building the 2N×144-byte combined bases buffer with
conditional point negation adds ~2% regression at n=100K.

The GLV module (`src/glv.rs`) is retained for potential future use if the aggregation
bottleneck is resolved (e.g. via signed-digit decomposition that reduces bucket count).

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
