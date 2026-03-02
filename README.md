# wgpu prover

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

# Open the generated trace in https://ui.perfetto.dev → load profile.json
```

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
| 2           | 85 ms   |
| 1K          | 266 ms  |
| 10K         | 690 ms  |
| 100K        | 2.47 s  |

### MSM batch (5 MSMs: a, b_g1, l, h, b_g2)

Isolated MSM batch — same 5-MSM workload as a real proof, with random bases and scalars.

| Points per MSM | Mean    |
|-----------------|---------|
| 100             | 165 ms  |
| 1K              | 259 ms  |
| 10K             | 642 ms  |
| 100K            | 2.10 s  |

### Bucket sorting (CPU)

Signed-digit scalar decomposition and bucket assignment (CPU-only, no GPU).

| Points | Mean    |
|--------|---------|
| 1K     | 2.38 ms |
| 10K    | 17.8 ms |
| 100K   | 139 ms  |

### MASP Sapling Output circuit

Real-world proof using the MASP Sapling Output circuit (31,211 constraints,
n=32,768).

| Phase                       | Time      |
|-----------------------------|-----------|
| Synthesis                   | 14.5 ms   |
| H polynomial (GPU)          | 152 ms    |
| Bucket sorting (CPU, GLV)   | 92.1 ms   |
| MSM batch (GPU)             | 781 ms    |
| **Total proof**             | **1.04 s** |

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

### 16. Signed-digit scalar decomposition with GLV sorting
Replaced unsigned window decomposition with signed-digit decomposition: each window
digit is in `[-(2^(c-1)-1), 2^(c-1)]` instead of `[0, 2^c-1]`, halving the number
of active buckets per window. When a digit is negative, the point is negated (cheap:
just negate Y coordinate) and assigned to the positive bucket. Combined with
structural cleanup: `BucketData` now carries `bucket_width`, and GLV-based sorting
was integrated.

### 17. G2 parallel subsum via complete addition formula
Replaced the single-threaded G2 running-sum (`@workgroup_size(1)`, 32 sequential
windows) with a parallel two-pass tree reduction matching G1's architecture. The
Metal shader compiler miscompiles `double_g2` under register pressure, so the
standard approach of `add_g2` (which calls `double_g2` for equal points) was blocked.

Solved by implementing the Renes-Costello-Batina (2015) complete addition formula
(`add_g2_complete`) in projective coordinates — a single branchless formula that
handles all cases (P+Q, P+P, P+(-P), P+O, O+P) without a separate doubling
function. This eliminates `double_g2` from the G2 subsum call graph entirely.

The aggregate phase keeps fast Jacobian mixed addition (unchanged); `weight_buckets_g2`
converts Jacobian→projective inline before `scalar_mul_g2`, then the parallel
`subsum_phase1_g2`/`subsum_phase2_g2` tree reduction sums the pre-weighted buckets.
- **sapling_output:** 16.9s → 16.6s (-1.8%)

### 18. GLV endomorphism for G1 MSMs
Integrated the pre-existing GLV (Gallant-Lambert-Vanstone) endomorphism optimization
into the proof MSM path. GLV decomposes each 255-bit scalar k into k1·P + k2·φ(P)
where k1, k2 are ~128-bit sub-scalars and φ(x,y) = (β·x, y) is a cheap endomorphism
(single Fq multiplication). This halves the number of Pippenger windows from ~20 to ~10
while doubling the point count (N → 2N), keeping aggregate work constant.

The endomorphism bases φ(P) are pre-computed in `PreparedProvingKey` (one-time cost),
and `compute_glv_bucket_sorting` handles GLV decomposition, conditional negation, and
signed-digit window decomposition for the ~128-bit sub-scalars.

At current point set sizes (~21K), the aggregate pass dominates and the window count
reduction has negligible impact. The optimization becomes more significant at larger
scales where weight+subsum represent a larger fraction of total GPU compute.
- **sapling_output:** 16.6s → 16.6s (neutral at ~21K points)

### 19. Sub-bucket chunking for workload-balanced aggregation
Real-world ZK circuits (e.g. Sapling Output) have highly skewed scalar distributions:
bucket value 1 in window 0 accumulates 11,767 points while the median bucket gets 1-2.
Since `aggregate_buckets` assigns 1 GPU thread per bucket, the thread handling the
overloaded bucket becomes the critical path — explaining why Sapling (32K points) took
16.6s while 100K uniform-random points took only 3.2s.

Sub-bucket chunking splits any bucket with >64 points into sub-buckets of at most 64
on the CPU, then adds a lightweight GPU reduce pass (`reduce_sub_buckets_g1/g2`) that
sums sub-bucket partial results back into per-bucket totals. The reduce pass uses
`add_g1_safe`/`add_g2_safe` (simple loop, no register pressure) and is dispatched only
when chunking is needed (`has_chunks` flag). Weight and subsum passes operate on the
original (non-chunked) bucket metadata, unchanged.

- **sapling_output:** 16.6s → 1.10s (15.1x speedup)
- **proof/n=100000:** 2.83s → 2.85s (neutral — uniform scalars trigger no chunking)

### 20. Parallel shared-memory G1 tree reduction (fix Metal threadgroup alignment)
The G1 tree reduction (`tree_reduction_ph1`) was the biggest bottleneck in the MSM
weight+subsum pipeline, accounting for ~43ms per G1 MSM pass. It ran with
`@workgroup_size(1)` — one thread per workgroup — because an earlier attempt at
parallel tree reduction using `var<workgroup> shared: array<PointG1, 64>` with
`@workgroup_size(64)` produced incorrect results on Metal.

The root cause was NOT a Metal GPU bug. The unpadded PointG1 struct (360 bytes) has
an array stride that is not 16-byte aligned (360 % 16 = 8). Metal's threadgroup memory
requires 16-byte-aligned strides. The fix: add `@size(128)` to each PointG1 struct
member in WGSL, padding each Fq coordinate from 120 to 128 bytes. This makes
PointG1 = 384 bytes (384 % 16 = 0), fixing the alignment for threadgroup arrays.

With the fix, the parallel tree reduction is restored: one workgroup of 64 threads per
window, with a 6-stage binary tree reduction in shared memory. Each thread sums a
strided subset of weighted buckets, then the tree reduction produces the final window sum.

- **msm_g1/n=100:** 28.0 ms → 21.7 ms (1.3x speedup)
- **msm_g1/n=1000:** 52.3 ms → 38.6 ms (1.35x speedup)
- **msm_g1/n=10000:** 125 ms → 99.6 ms (1.26x speedup)
- **msm_g1/n=100000:** 367 ms → 341 ms (1.08x speedup)
- **proof/n=1000:** 266 ms → 234 ms (1.14x speedup)
- **proof/n=10000:** 690 ms → 612 ms (1.13x speedup)
- **sapling_output:** 1.04s → 1.02s (1.02x speedup)

### 21. Two-phase MSM submission (overlap GPU MSM with CPU h bucket sorting)

Previously, all 5 MSMs (a, b1, l, h, b2) were submitted together in a single batch after
h bucket sorting completed. This left the GPU idle for ~38ms while the CPU computed h bucket
data and enqueued all 5 MSMs. By splitting the batch into two phases — submitting a/b1/l/b2
immediately after h_poly read, then h after its bucket sorting — the GPU begins processing
the first 4 MSMs while the CPU computes h bucket data in parallel.

Also refactored `gpu_msm_batch_bytes` into reusable `enqueue_msm_g1`, `enqueue_msm_g2`, and
`readback_msms` functions for more flexible MSM scheduling.

- **sapling_output:** 997 ms → 982 ms (~1.5% improvement)

### OPT-22: Persistent GPU bases across proofs

Pre-upload and convert base point buffers to the GPU once via `GpuProvingKey`, then reuse
across all proofs for the same circuit. Eliminates ~102 MB of per-proof base uploads and
5 `to_montgomery` GPU dispatches.

The key challenge was that `compute_glv_bucket_sorting` baked per-proof GLV negation into
the bases buffer, making it proof-dependent. Solved by folding GLV negation into the
`base_indices` sign bit (XOR with signed-digit window sign) via a new
`compute_glv_bucket_data` function, making the interleaved bases buffer circuit-fixed.

One-time setup cost: ~39ms for upload + Montgomery conversion (amortized across proofs).

- **sapling_output:** 982 ms → 965 ms (~1.7% improvement per proof)
  - msm enqueue a/b1/l/b2: 15.9ms → 2.3ms (no base upload)
  - bucket sorting: 57.6ms → 45.0ms (no combined_bases building)

### OPT-23: Adaptive bucket width (c) per MSM

Replace the fixed bucket width `c=13` for all G1 MSMs with a per-MSM optimal value
computed by `optimal_glv_c(n)`. The function minimizes the Pippenger cost
`f(c) = ceil(128/c) × (2n + 2^(c-1))` over c ∈ [10, 13], capped at 13 because
values above 13 cause exponential subsum cost growth on GPU (2^(c-1)/64 sequential
additions per thread in tree reduction).

Also increased `scalar_mul_g1`/`scalar_mul_g2` loop bounds from 14 to 16 bits for
future-proofing (early `break` ensures zero cost for small c values).

- **sapling_output:** 965 ms → 987 ms (no change — all MSMs select c=13 at n~21-31K)
- **msm_g1/n=1000:** 38.6 ms → 33.0 ms (**-14.5%**, c=10 selected)
- **proof/n=2:** 86.6 ms → 80.1 ms (**-7.5%**, c=10 selected for small MSMs)

## Latest Benchmark Results

Measured on Apple M3 Max. Criterion median times.

| Benchmark | Time |
|---|---|
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

### G1 running-sum subsum (replace weight + tree reduction)
**Idea:** Replace the G1 `weight_buckets_g1` + `subsum_phase1_g1` + `subsum_phase2_g1`
three-pass pipeline with a single Pippenger running-sum pass (matching the G2 approach).
The running-sum walks active buckets in descending order, implicitly weighting each bucket
by its value — eliminating the need for `scalar_mul_g1` entirely. Total work drops ~10x
(~164K additions vs ~1.56M EC ops for weight+subsum combined).

**Why discarded:** The running-sum dispatches only `num_windows` = 20 single-threaded
workgroups, compared to the current approach's ~82K threads across weight (1280 WGs × 64)
and phase1 (640 WGs × 1). Despite doing ~10x less total work, the GPU is severely
underutilized with only 20 active threads — a +18% regression at the Sapling Output
circuit (19.66s vs 16.65s readback). This confirms that `aggregate_buckets_g1` dominates
MSM cost and the weight+subsum passes are efficiently parallelized. Future optimization
should target aggregate_buckets directly (e.g., reducing per-addition cost or improving
GPU occupancy).

### Suffix scan + gap-weight (Pippenger running-sum via parallel scan)
**Idea:** Replace the `weight_buckets` scalar multiplication pass with a three-phase
parallel suffix scan followed by a gap-weighted multiplication. Uses the Pippenger
identity `sum(v * B[v]) = sum(gap_j * R[j])` where `R[j]` is the suffix sum of bucket
points and `gap_j = v_j - v_{j-1}`. Since gaps are typically small (avg ~8 = 3 bits),
the `scalar_mul` per bucket drops from ~11-bit (22ms) to ~3-bit (0.8ms) average.

**Why discarded:** While the gap-weight pass was 27x faster than the original weight
pass (0.8ms vs 22ms), the three-phase suffix scan itself costs ~55ms of additional point
additions (43ms phase1 + 10ms phase2 + 1.6ms phase3). The suffix scan is O(N) point
additions — the same cost as the tree reduction subsum it runs alongside. This results
in two O(N) addition passes instead of one, producing a net regression of +33ms
(114ms total vs 81ms original for weight+subsum). GPU profiling confirmed:

| Pass | Old (ms) | New (ms) |
|------|----------|----------|
| weight / gap_weight | 22.0 | 0.8 |
| suffix_scan (3 phases) | — | 55.0 |
| tree_reduction_ph1 | 43.0 | 43.0 |
| **Total** | **65.0** | **98.8** |

The fundamental limitation is that point additions dominate GPU compute cost, and any
approach that adds an O(N) scan pass before the existing O(N) tree reduction cannot be
faster — regardless of how cheap the scalar multiplication becomes. (Note: the
`@workgroup_size(1)` constraint has since been resolved by OPT-20.)

### Intra-bucket parallel tree reduction for aggregate_buckets_g1
**Idea:** Replace the sequential per-thread bucket aggregation (one thread iterates over
all points in a sub-bucket) with a parallel tree reduction: one workgroup of 64 threads
per sub-bucket, each thread loads one point, then 6-stage binary tree reduction in shared
memory (`var<workgroup> agg_shared_g1: array<PointG1, 64>`, 24KB). This mirrors the
existing `subsum_phase1_g1` pattern and reduces per-bucket latency from O(bucket_size)
sequential additions to O(log2(64))=6 parallel levels.

**Why discarded:** Caused a +73% regression (982ms → 1.70s). Three compounding problems:

1. **64x workgroup explosion:** Changed from 605 workgroups (38,701 threads total) to
   38,701 workgroups (2.5M threads). Most buckets have only 3-16 points (mean=16.1 for
   h MSM), so 48-61 of 64 threads per workgroup are completely idle, loading identity
   and participating in barriers for no useful work.

2. **Shared memory pressure:** 24KB per workgroup (64 × 384 bytes) limits GPU occupancy
   to ~1 workgroup per compute unit on Metal (32KB threadgroup limit), causing massive
   serialization of the 38,701 workgroups.

3. **Barrier overhead:** 6 `workgroupBarrier()` calls per bucket × 38,701 workgroups.
   For small buckets (size 3-16), the barrier synchronization cost dominates the actual
   addition work.

The approach works well for `subsum_phase1_g1` because it dispatches only ~10 workgroups
(one per window), but fails catastrophically when dispatching ~38,700 workgroups for
per-bucket parallelism.
