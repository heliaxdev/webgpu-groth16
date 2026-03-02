//! Pippenger bucket sorting with signed-digit scalar decomposition.
//!
//! Prepares [`BucketData`] for GPU MSM dispatch by decomposing scalars into
//! signed-digit windows and grouping points by (window, bucket_value).
//! Large buckets are split into sub-buckets for GPU load balancing (see
//! [`GpuCurve::MSM_MAX_CHUNK_SIZE`]).
//!
//! Two modes (selected by [`GpuCurve::HAS_G1_GLV`]):
//! - **Standard** ([`compute_bucket_sorting`]): direct scalar decomposition
//! - **GLV-capable** ([`compute_glv_bucket_sorting`], [`compute_glv_bucket_data`]):
//!   uses curve-provided endomorphism decomposition hooks when available

use crate::gpu::curve::{G1MsmDecomposition, GpuCurve};

/// Bucket sorting result for GPU MSM dispatch.
///
/// Uses a Structure-of-Arrays layout: each array is uploaded as a separate
/// `storage<read_only>` GPU buffer. This avoids struct padding issues in WGSL
/// and allows independent buffer bindings per kernel.
///
/// When sub-bucket chunking is active (`has_chunks == true`), the parallel arrays
/// (`bucket_pointers`, `bucket_sizes`, `bucket_values`, `window_starts`,
/// `window_counts`) describe *sub-buckets* (dispatched units), not logical buckets.
/// The `reduce_starts`/`reduce_counts` arrays map original bucket indices to their
/// sub-bucket ranges for a post-aggregation reduction pass.
///
/// Invariants:
/// - `bucket_pointers[i]` is the starting index in `base_indices` for sub-bucket `i`
/// - `bucket_sizes[i]` is the count of points in sub-bucket `i`
/// - `bucket_values[i]` is the scalar weight for sub-bucket `i` (in `[1, 2^(c-1)]`)
/// - `window_starts[w]` is the first sub-bucket index belonging to window `w`
/// - `window_counts[w]` is the number of sub-buckets in window `w`
/// - `reduce_starts[j]` is the first sub-bucket index for original bucket `j`
/// - `reduce_counts[j]` is the number of sub-buckets for original bucket `j`
pub struct BucketData {
    pub base_indices: Vec<u32>,
    /// Sub-bucket pointers into base_indices (length = num_dispatched).
    pub bucket_pointers: Vec<u32>,
    /// Sub-bucket sizes (length = num_dispatched).
    pub bucket_sizes: Vec<u32>,
    /// Sub-bucket values, same as parent's value (length = num_dispatched).
    pub bucket_values: Vec<u32>,
    /// Sub-bucket window starts (length = num_windows).
    pub window_starts: Vec<u32>,
    /// Sub-bucket counts per window (length = num_windows).
    pub window_counts: Vec<u32>,
    pub num_windows: u32,
    /// Number of original (logical) buckets.
    pub num_active_buckets: u32,
    /// Number of dispatched sub-buckets (>= num_active_buckets when chunking occurs).
    pub num_dispatched: u32,
    /// Original bucket values for weight/subsum passes (length = num_active_buckets).
    pub orig_bucket_values: Vec<u32>,
    /// Original window starts for weight/subsum passes (length = num_windows).
    pub orig_window_starts: Vec<u32>,
    /// Original window counts for weight/subsum passes (length = num_windows).
    pub orig_window_counts: Vec<u32>,
    /// Start offset in the dispatch buffer for each original bucket.
    pub reduce_starts: Vec<u32>,
    /// Number of sub-buckets for each original bucket.
    pub reduce_counts: Vec<u32>,
    /// Whether any bucket was split into sub-buckets.
    pub has_chunks: bool,
    pub bucket_width: usize,
}

impl BucketData {
    /// Print bucket size distribution statistics for diagnosing workload imbalance.
    /// Only active when the `timing` feature is enabled.
    #[cfg(feature = "timing")]
    pub fn print_distribution_stats(&self, label: &str) {
        if self.num_active_buckets == 0 {
            eprintln!("[bucket-diag] {label}: 0 active buckets");
            return;
        }
        let mut sizes: Vec<u32> = self.bucket_sizes.clone();
        sizes.sort();
        let n = sizes.len();
        let total: u32 = sizes.iter().sum();
        let max = *sizes.last().unwrap();
        let min = *sizes.first().unwrap();
        let mean = total as f64 / n as f64;
        let median = sizes[n / 2];
        let p90 = sizes[(n * 90) / 100];
        let p95 = sizes[(n * 95) / 100];
        let p99 = sizes[n.saturating_sub(1).min((n * 99) / 100)];

        let over_64 = sizes.iter().filter(|&&s| s > 64).count();
        let over_256 = sizes.iter().filter(|&&s| s > 256).count();
        let over_1024 = sizes.iter().filter(|&&s| s > 1024).count();

        eprintln!(
            "[bucket-diag] {label}: {n} active buckets, {total} total points, c={}",
            self.bucket_width
        );
        eprintln!("[bucket-diag]   min={min} max={max} mean={mean:.1} median={median}");
        eprintln!("[bucket-diag]   p90={p90} p95={p95} p99={p99}");
        eprintln!("[bucket-diag]   >64: {over_64}  >256: {over_256}  >1024: {over_1024}");

        // Per-window summary for windows with large buckets
        for w in 0..self.num_windows as usize {
            let start = self.window_starts[w] as usize;
            let count = self.window_counts[w] as usize;
            if count == 0 {
                continue;
            }
            let w_sizes: Vec<u32> = (start..start + count)
                .map(|i| self.bucket_sizes[i])
                .collect();
            let w_max = *w_sizes.iter().max().unwrap();
            let w_total: u32 = w_sizes.iter().sum();
            // Find the bucket value with max size
            let max_idx = w_sizes.iter().position(|&s| s == w_max).unwrap();
            let max_val = self.bucket_values[start + max_idx];
            if w_max > 32 {
                eprintln!(
                    "[bucket-diag]   window {w}: {count} buckets, max_size={w_max} (val={max_val}), total={w_total}"
                );
            }
        }
    }
}

/// Builds `BucketData` from pre-computed signed-digit window decompositions.
///
/// `all_windows[i]` contains the (absolute_value, is_negative) pairs for point `i`.
/// `c` is the bucket width (window size in bits).
///
/// ## Algorithm (two-pass Pippenger bucket sorting with sub-bucket chunking)
///
/// **Pass 1 — Group points by (window, bucket_value):**
/// For each window w, iterate over all points and place each into the bucket
/// corresponding to its signed-digit value. Produces flat arrays of:
/// base_indices (point IDs, sign-encoded), pointers, sizes, and values per bucket.
///
/// **Pass 2 — Split oversized buckets for GPU load balancing:**
/// Buckets with more than `G::MSM_MAX_CHUNK_SIZE` points are split into sub-buckets.
/// Each sub-bucket becomes an independent GPU thread. A reduce_starts/reduce_counts
/// table records which sub-buckets belong to the same logical bucket, so a later
/// GPU reduce pass can sum the sub-bucket partials back together.
fn build_bucket_data<G: GpuCurve>(all_windows: &[Vec<(u32, bool)>], c: usize) -> BucketData {
    let num_windows = all_windows.iter().map(|w| w.len()).max().unwrap_or(0);
    let num_buckets = (1usize << (c - 1)) + 1;

    // First pass: collect points into logical buckets per window.
    let mut base_indices = Vec::new();
    let mut orig_pointers = Vec::new();
    let mut orig_sizes = Vec::new();
    let mut orig_values = Vec::new();
    let mut orig_window_starts = Vec::new();
    let mut orig_window_counts = Vec::new();

    for w in 0..num_windows {
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); num_buckets];

        for (i, windows) in all_windows.iter().enumerate() {
            if w < windows.len() {
                let (abs, neg) = windows[w];
                if abs != 0 {
                    let entry = if neg {
                        i as u32 | G::MSM_INDEX_SIGN_BIT
                    } else {
                        i as u32
                    };
                    buckets[abs as usize].push(entry);
                }
            }
        }

        orig_window_starts.push(orig_values.len() as u32);
        let mut count = 0u32;

        for (val, indices) in buckets.into_iter().enumerate() {
            if !indices.is_empty() {
                orig_pointers.push(base_indices.len() as u32);
                orig_sizes.push(indices.len() as u32);
                orig_values.push(val as u32);
                base_indices.extend(indices);
                count += 1;
            }
        }
        orig_window_counts.push(count);
    }

    let num_active_buckets = orig_sizes.len() as u32;

    // Second pass: split large buckets into sub-buckets.
    let mut bucket_pointers = Vec::new();
    let mut bucket_sizes = Vec::new();
    let mut bucket_values = Vec::new();
    let mut window_starts = Vec::new();
    let mut window_counts = Vec::new();
    let mut reduce_starts = Vec::new();
    let mut reduce_counts = Vec::new();
    let mut has_chunks = false;

    for w in 0..num_windows {
        let w_start = orig_window_starts[w] as usize;
        let w_count = orig_window_counts[w] as usize;
        window_starts.push(bucket_pointers.len() as u32);
        let mut dispatched_in_window = 0u32;

        for b in 0..w_count {
            let orig_idx = w_start + b;
            let ptr = orig_pointers[orig_idx];
            let size = orig_sizes[orig_idx];
            let val = orig_values[orig_idx];

            let sub_start = bucket_pointers.len() as u32;

            if size <= G::MSM_MAX_CHUNK_SIZE {
                bucket_pointers.push(ptr);
                bucket_sizes.push(size);
                bucket_values.push(val);
                reduce_starts.push(sub_start);
                reduce_counts.push(1);
                dispatched_in_window += 1;
            } else {
                has_chunks = true;
                let num_chunks = size.div_ceil(G::MSM_MAX_CHUNK_SIZE);
                for chunk in 0..num_chunks {
                    let chunk_start = ptr + chunk * G::MSM_MAX_CHUNK_SIZE;
                    let chunk_size =
                        (size - chunk * G::MSM_MAX_CHUNK_SIZE).min(G::MSM_MAX_CHUNK_SIZE);
                    bucket_pointers.push(chunk_start);
                    bucket_sizes.push(chunk_size);
                    bucket_values.push(val);
                    dispatched_in_window += 1;
                }
                reduce_starts.push(sub_start);
                reduce_counts.push(num_chunks);
            }
        }
        window_counts.push(dispatched_in_window);
    }

    let num_dispatched = bucket_pointers.len() as u32;

    BucketData {
        base_indices,
        bucket_pointers,
        bucket_sizes,
        bucket_values,
        window_starts,
        window_counts,
        num_windows: num_windows as u32,
        num_active_buckets,
        num_dispatched,
        orig_bucket_values: orig_values,
        orig_window_starts,
        orig_window_counts,
        reduce_starts,
        reduce_counts,
        has_chunks,
        bucket_width: c,
    }
}

pub fn optimal_glv_c<G: GpuCurve>(n: usize) -> usize {
    G::g1_msm_bucket_width(n)
}

pub fn compute_bucket_sorting<G: GpuCurve>(scalars: &[G::Scalar]) -> BucketData {
    compute_bucket_sorting_with_width::<G>(scalars, G::bucket_width())
}

pub fn compute_bucket_sorting_with_width<G: GpuCurve>(
    scalars: &[G::Scalar],
    c: usize,
) -> BucketData {
    let all_windows: Vec<Vec<(u32, bool)>> = scalars
        .iter()
        .map(|s| G::scalar_to_signed_windows(s, c))
        .collect();
    build_bucket_data::<G>(&all_windows, c)
}

/// Curve-capability-aware G1 bucket sorting with signed-digit decomposition.
///
/// For GLV-capable curves, decomposes each scalar into two components and builds a
/// 2N-entry bases buffer with conditional point negation. For non-GLV curves,
/// falls back to standard signed-window sorting and returns the original base bytes.
///
/// Returns `(combined_bases_bytes, bucket_data)` where `combined_bases_bytes` is
/// a 2N×G1_GPU_BYTES buffer laid out as:
///   [maybe_neg(P₀), maybe_neg(φ(P₀)), maybe_neg(P₁), maybe_neg(φ(P₁)), ...]
pub fn compute_glv_bucket_sorting<G: GpuCurve>(
    scalars: &[G::Scalar],
    bases_bytes: &[u8],
    phi_bases_bytes: &[u8],
    c: usize,
) -> (Vec<u8>, BucketData) {
    if !G::HAS_G1_GLV {
        let bd = compute_bucket_sorting_with_width::<G>(scalars, c);
        return (bases_bytes.to_vec(), bd);
    }

    let n = scalars.len();
    debug_assert_eq!(bases_bytes.len(), n * G::G1_GPU_BYTES);
    debug_assert_eq!(phi_bases_bytes.len(), n * G::G1_GPU_BYTES);

    // Decompose all scalars and build the combined bases buffer.
    let mut combined_bases = Vec::with_capacity(n * 2 * G::G1_GPU_BYTES);
    let mut all_windows: Vec<Vec<(u32, bool)>> = Vec::with_capacity(n * 2);

    for (i, scalar) in scalars.iter().enumerate() {
        if let Some((k1_windows, k1_neg, k2_windows, k2_neg)) =
            G::decompose_g1_msm_scalar_glv_windows(scalar, c)
        {
            let src_start = i * G::G1_GPU_BYTES;
            let mut p_bytes = bases_bytes[src_start..src_start + G::G1_GPU_BYTES].to_vec();
            if k1_neg {
                G::negate_g1_base_bytes(&mut p_bytes);
            }
            combined_bases.extend_from_slice(&p_bytes);

            let mut phi_bytes = phi_bases_bytes[src_start..src_start + G::G1_GPU_BYTES].to_vec();
            if k2_neg {
                G::negate_g1_base_bytes(&mut phi_bytes);
            }
            combined_bases.extend_from_slice(&phi_bytes);

            all_windows.push(k1_windows);
            all_windows.push(k2_windows);
        } else if let G1MsmDecomposition::Standard { windows } =
            G::decompose_g1_msm_scalar(scalar, c)
        {
            let src_start = i * G::G1_GPU_BYTES;
            combined_bases.extend_from_slice(&bases_bytes[src_start..src_start + G::G1_GPU_BYTES]);
            all_windows.push(windows);
        }
    }

    (combined_bases, build_bucket_data::<G>(&all_windows, c))
}

/// Curve-capability-aware bucket sorting that returns only BucketData (no bases buffer).
///
/// For GLV-capable curves with persistent bases, GLV negation is folded into
/// `base_indices` sign bits (XOR with signed-digit window sign) instead of mutating
/// base bytes. For non-GLV curves this is equivalent to standard sorting.
pub fn compute_glv_bucket_data<G: GpuCurve>(scalars: &[G::Scalar], c: usize) -> BucketData {
    if !G::HAS_G1_GLV {
        return compute_bucket_sorting_with_width::<G>(scalars, c);
    }

    let n = scalars.len();
    let mut all_windows: Vec<Vec<(u32, bool)>> = Vec::with_capacity(n * 2);

    for scalar in scalars.iter() {
        if let Some((mut k1_windows, k1_neg, mut k2_windows, k2_neg)) =
            G::decompose_g1_msm_scalar_glv_windows(scalar, c)
        {
            if k1_neg {
                for w in &mut k1_windows {
                    if w.0 != 0 {
                        w.1 = !w.1;
                    }
                }
            }
            all_windows.push(k1_windows);

            if k2_neg {
                for w in &mut k2_windows {
                    if w.0 != 0 {
                        w.1 = !w.1;
                    }
                }
            }
            all_windows.push(k2_windows);
        } else if let G1MsmDecomposition::Standard { windows } =
            G::decompose_g1_msm_scalar(scalar, c)
        {
            all_windows.push(windows);
        }
    }

    build_bucket_data::<G>(&all_windows, c)
}

#[cfg(test)]
mod tests;
