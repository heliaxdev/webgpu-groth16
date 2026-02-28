use crate::glv;
use crate::gpu::curve::GpuCurve;

pub struct BucketData {
    pub base_indices: Vec<u32>,
    pub bucket_pointers: Vec<u32>,
    pub bucket_sizes: Vec<u32>,
    pub bucket_values: Vec<u32>,
    pub window_starts: Vec<u32>,
    pub window_counts: Vec<u32>,
    pub num_windows: u32,
    pub num_active_buckets: u32,
}

/// Sign bit used to encode point negation in base_indices entries.
/// When this bit is set, the GPU negates the point's y-coordinate before adding.
const SIGN_BIT: u32 = 1 << 31;

pub fn compute_bucket_sorting<G: GpuCurve>(scalars: &[G::Scalar]) -> BucketData {
    compute_bucket_sorting_with_width::<G>(scalars, G::bucket_width())
}

pub fn compute_bucket_sorting_with_width<G: GpuCurve>(
    scalars: &[G::Scalar],
    c: usize,
) -> BucketData {
    // Pre-compute all signed scalar window decompositions.
    // Each window is (absolute_value, is_negative).
    let all_windows: Vec<Vec<(u32, bool)>> = scalars
        .iter()
        .map(|s| G::scalar_to_signed_windows(s, c))
        .collect();

    // The number of windows is determined by the longest decomposition
    // (may be num_windows+1 if final carry produced an extra window).
    let num_windows = all_windows.iter().map(|w| w.len()).max().unwrap_or(0);

    // Bucket values now range 1..2^(c-1) (halved from unsigned).
    let num_buckets = (1usize << (c - 1)) + 1;

    let mut base_indices = Vec::new();
    let mut bucket_pointers = Vec::new();
    let mut bucket_sizes = Vec::new();
    let mut bucket_values = Vec::new();
    let mut window_starts = Vec::new();
    let mut window_counts = Vec::new();

    for w in 0..num_windows {
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); num_buckets];

        for (i, windows) in all_windows.iter().enumerate() {
            if w < windows.len() {
                let (abs, neg) = windows[w];
                if abs != 0 {
                    // Encode sign in MSB of the base index
                    let entry = if neg { i as u32 | SIGN_BIT } else { i as u32 };
                    buckets[abs as usize].push(entry);
                }
            }
        }

        window_starts.push(bucket_values.len() as u32);
        let mut count = 0;

        for (val, indices) in buckets.into_iter().enumerate() {
            if !indices.is_empty() {
                bucket_pointers.push(base_indices.len() as u32);
                bucket_sizes.push(indices.len() as u32);
                bucket_values.push(val as u32);
                base_indices.extend(indices);
                count += 1;
            }
        }
        window_counts.push(count);
    }

    let num_active_buckets = bucket_sizes.len() as u32;

    BucketData {
        base_indices,
        bucket_pointers,
        bucket_sizes,
        bucket_values,
        window_starts,
        window_counts,
        num_windows: num_windows as u32,
        num_active_buckets,
    }
}

/// GLV-aware bucket sorting for G1 MSM.
///
/// Decomposes each scalar via GLV into two ~128-bit components, builds a 2N-entry
/// bases buffer with conditional point negation, and produces BucketData for 9
/// windows (128-bit scalars with c=15) over the 2N points.
///
/// Returns `(combined_bases_bytes, bucket_data)` where `combined_bases_bytes` is
/// a 2N×144-byte buffer laid out as:
///   [maybe_neg(P₀), maybe_neg(φ(P₀)), maybe_neg(P₁), maybe_neg(φ(P₁)), ...]
pub fn compute_glv_bucket_sorting<G: GpuCurve>(
    scalars: &[G::Scalar],
    bases_bytes: &[u8],
    phi_bases_bytes: &[u8],
    c: usize,
) -> (Vec<u8>, BucketData) {
    let n = scalars.len();
    debug_assert_eq!(bases_bytes.len(), n * 144);
    debug_assert_eq!(phi_bases_bytes.len(), n * 144);

    let num_windows = 128_usize.div_ceil(c);

    // Decompose all scalars and build the combined bases buffer.
    let mut combined_bases = Vec::with_capacity(n * 2 * 144);
    let mut all_windows: Vec<Vec<u32>> = Vec::with_capacity(n * 2);

    for i in 0..n {
        let (k1, k1_neg, k2, k2_neg) = glv::glv_decompose(&scalars[i]);

        // Entry 2i: original base P_i (conditionally negated)
        let src_start = i * 144;
        let mut p_bytes = bases_bytes[src_start..src_start + 144].to_vec();
        if k1_neg {
            glv::negate_g1_bytes(&mut p_bytes);
        }
        combined_bases.extend_from_slice(&p_bytes);

        // Entry 2i+1: endomorphism base φ(P_i) (conditionally negated)
        let mut phi_bytes = phi_bases_bytes[src_start..src_start + 144].to_vec();
        if k2_neg {
            glv::negate_g1_bytes(&mut phi_bytes);
        }
        combined_bases.extend_from_slice(&phi_bytes);

        // Window decompositions for the two half-scalars
        all_windows.push(glv::u128_to_windows(k1, c));
        all_windows.push(glv::u128_to_windows(k2, c));
    }

    // Run standard bucket sorting on the 2N window decompositions.
    let mut base_indices = Vec::new();
    let mut bucket_pointers = Vec::new();
    let mut bucket_sizes = Vec::new();
    let mut bucket_values = Vec::new();
    let mut window_starts = Vec::new();
    let mut window_counts = Vec::new();

    for w in 0..num_windows {
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); 1 << c];

        for (i, windows) in all_windows.iter().enumerate() {
            if w < windows.len() {
                let val = windows[w] as usize;
                if val != 0 {
                    buckets[val].push(i as u32);
                }
            }
        }

        window_starts.push(bucket_values.len() as u32);
        let mut count = 0;

        for (val, indices) in buckets.into_iter().enumerate() {
            if !indices.is_empty() {
                bucket_pointers.push(base_indices.len() as u32);
                bucket_sizes.push(indices.len() as u32);
                bucket_values.push(val as u32);
                base_indices.extend(indices);
                count += 1;
            }
        }
        window_counts.push(count);
    }

    let num_active_buckets = bucket_sizes.len() as u32;

    let bd = BucketData {
        base_indices,
        bucket_pointers,
        bucket_sizes,
        bucket_values,
        window_starts,
        window_counts,
        num_windows: num_windows as u32,
        num_active_buckets,
    };

    (combined_bases, bd)
}
