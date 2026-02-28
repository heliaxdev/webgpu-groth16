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

#[cfg(test)]
mod tests {
    use super::*;
    use blstrs::Bls12;
    use ff::Field;
    use rand_core::OsRng;

    type Scalar = <Bls12 as GpuCurve>::Scalar;

    const SIGN_BIT_MASK: u32 = 1 << 31;
    const INDEX_MASK: u32 = !SIGN_BIT_MASK;

    /// Verify all structural invariants of BucketData.
    fn assert_bucket_data_invariants(bd: &BucketData, n: usize, c: usize) {
        let half = 1u32 << (c - 1);

        // Parallel array lengths
        assert_eq!(bd.bucket_pointers.len(), bd.num_active_buckets as usize);
        assert_eq!(bd.bucket_sizes.len(), bd.num_active_buckets as usize);
        assert_eq!(bd.bucket_values.len(), bd.num_active_buckets as usize);
        assert_eq!(bd.window_starts.len(), bd.num_windows as usize);
        assert_eq!(bd.window_counts.len(), bd.num_windows as usize);

        // Sum of window_counts == num_active_buckets
        let total_active: u32 = bd.window_counts.iter().sum();
        assert_eq!(total_active, bd.num_active_buckets);

        // Sum of bucket_sizes == base_indices.len()
        let total_entries: u32 = bd.bucket_sizes.iter().sum();
        assert_eq!(total_entries as usize, bd.base_indices.len());

        // Every active bucket is non-empty and has valid pointers
        for i in 0..bd.num_active_buckets as usize {
            assert!(bd.bucket_sizes[i] > 0, "empty bucket at index {i}");
            let ptr = bd.bucket_pointers[i] as usize;
            let end = ptr + bd.bucket_sizes[i] as usize;
            assert!(
                end <= bd.base_indices.len(),
                "bucket {i} overflows base_indices"
            );
        }

        // Bucket values are in [1, 2^(c-1)]
        for (i, &v) in bd.bucket_values.iter().enumerate() {
            assert!(v >= 1, "bucket value 0 at index {i}");
            assert!(
                v <= half,
                "bucket value {v} exceeds half={half} at index {i}"
            );
        }

        // Within each window, bucket values are in ascending order
        for w in 0..bd.num_windows as usize {
            let start = bd.window_starts[w] as usize;
            let count = bd.window_counts[w] as usize;
            for j in 1..count {
                assert!(
                    bd.bucket_values[start + j] > bd.bucket_values[start + j - 1],
                    "bucket values not sorted in window {w}"
                );
            }
        }

        // All base indices (lower 31 bits) are valid scalar indices
        for &raw in &bd.base_indices {
            let idx = raw & INDEX_MASK;
            assert!(
                (idx as usize) < n,
                "base index {idx} out of range for n={n}"
            );
        }
    }

    /// Verify that every non-zero signed window digit appears in exactly the right bucket.
    fn assert_bucket_data_covers_all_windows(bd: &BucketData, scalars: &[Scalar], c: usize) {
        let n = scalars.len();
        let all_windows: Vec<Vec<(u32, bool)>> = scalars
            .iter()
            .map(|s| <Bls12 as GpuCurve>::scalar_to_signed_windows(s, c))
            .collect();

        // For each window, collect what bucket sorting produced.
        for w in 0..bd.num_windows as usize {
            let w_start = bd.window_starts[w] as usize;
            let w_count = bd.window_counts[w] as usize;

            // Build a map: (scalar_index, sign) -> bucket_value from the BucketData.
            let mut found: std::collections::HashMap<(u32, bool), u32> =
                std::collections::HashMap::new();
            for b in 0..w_count {
                let bucket_idx = w_start + b;
                let val = bd.bucket_values[bucket_idx];
                let ptr = bd.bucket_pointers[bucket_idx] as usize;
                let size = bd.bucket_sizes[bucket_idx] as usize;
                for k in 0..size {
                    let raw = bd.base_indices[ptr + k];
                    let idx = raw & INDEX_MASK;
                    let neg = (raw & SIGN_BIT_MASK) != 0;
                    let prev = found.insert((idx, neg), val);
                    assert!(
                        prev.is_none(),
                        "scalar {idx} appears twice in window {w}"
                    );
                }
            }

            // Verify every non-zero signed window digit was placed in the correct bucket.
            for i in 0..n {
                if w < all_windows[i].len() {
                    let (abs, neg) = all_windows[i][w];
                    if abs != 0 {
                        let got = found.get(&(i as u32, neg));
                        assert_eq!(
                            got,
                            Some(&abs),
                            "scalar {i} window {w}: expected bucket val={abs} neg={neg}, got {:?}",
                            got
                        );
                    } else {
                        // Zero windows should not appear in any bucket.
                        assert!(
                            !found.contains_key(&(i as u32, false))
                                && !found.contains_key(&(i as u32, true)),
                            "scalar {i} has zero window {w} but appeared in bucket data"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bucket_sorting_structural_invariants_random() {
        let c = <Bls12 as GpuCurve>::bucket_width();
        for &n in &[1, 2, 10, 100, 1000] {
            let scalars: Vec<Scalar> = (0..n).map(|_| Scalar::random(OsRng)).collect();
            let bd = compute_bucket_sorting::<Bls12>(&scalars);
            assert_bucket_data_invariants(&bd, n, c);
            assert_bucket_data_covers_all_windows(&bd, &scalars, c);
        }
    }

    #[test]
    fn bucket_sorting_structural_invariants_g2_width() {
        let c = <Bls12 as GpuCurve>::g2_bucket_width();
        let scalars: Vec<Scalar> = (0..200).map(|_| Scalar::random(OsRng)).collect();
        let bd = compute_bucket_sorting_with_width::<Bls12>(&scalars, c);
        assert_bucket_data_invariants(&bd, 200, c);
        assert_bucket_data_covers_all_windows(&bd, &scalars, c);
    }

    #[test]
    fn bucket_sorting_all_zero_scalars() {
        let c = <Bls12 as GpuCurve>::bucket_width();
        let scalars = vec![Scalar::ZERO; 10];
        let bd = compute_bucket_sorting::<Bls12>(&scalars);
        // Zero scalars produce all-zero windows -> no buckets at all.
        assert_eq!(bd.base_indices.len(), 0);
        assert_eq!(bd.num_active_buckets, 0);
        assert_bucket_data_invariants(&bd, 10, c);
    }

    #[test]
    fn bucket_sorting_single_scalar() {
        let c = <Bls12 as GpuCurve>::bucket_width();
        let scalars = vec![Scalar::from(42u64)];
        let bd = compute_bucket_sorting::<Bls12>(&scalars);
        assert_bucket_data_invariants(&bd, 1, c);
        assert_bucket_data_covers_all_windows(&bd, &scalars, c);
        // Scalar 42 fits in one window (42 < 2^12), no sign bit needed.
        // So it should produce exactly 1 active bucket.
        assert_eq!(bd.num_active_buckets, 1);
        assert_eq!(bd.bucket_values[0], 42);
        assert_eq!(bd.base_indices[0] & INDEX_MASK, 0); // scalar index 0
        assert_eq!(bd.base_indices[0] & SIGN_BIT_MASK, 0); // not negated
    }

    #[test]
    fn bucket_sorting_scalar_needing_negation() {
        // A scalar whose first window is >= 2^(c-1) should produce a negative entry.
        let c = <Bls12 as GpuCurve>::bucket_width();
        let half = 1u64 << (c - 1);
        // Scalar = 2^(c-1) should produce: window 0 = -(2^c - 2^(c-1)) = -2^(c-1), carry=1
        // i.e., (abs=half, neg=true) for window 0, and (1, false) for window 1.
        let s = Scalar::from(half);
        let scalars = vec![s];
        let bd = compute_bucket_sorting::<Bls12>(&scalars);
        assert_bucket_data_invariants(&bd, 1, c);
        assert_bucket_data_covers_all_windows(&bd, &scalars, c);

        // Verify at least one entry has the sign bit set.
        let has_neg = bd.base_indices.iter().any(|&r| r & SIGN_BIT_MASK != 0);
        assert!(
            has_neg,
            "scalar 2^(c-1) should produce a negated window entry"
        );
    }

    #[test]
    fn bucket_sorting_identical_scalars() {
        let c = <Bls12 as GpuCurve>::bucket_width();
        let s = Scalar::from(7u64);
        let scalars = vec![s; 50];
        let bd = compute_bucket_sorting::<Bls12>(&scalars);
        assert_bucket_data_invariants(&bd, 50, c);
        assert_bucket_data_covers_all_windows(&bd, &scalars, c);

        // All 50 scalars have the same window decomposition, so each active bucket
        // should contain exactly 50 entries.
        for &size in &bd.bucket_sizes {
            assert_eq!(size, 50);
        }
    }

    #[test]
    fn bucket_sorting_max_scalar() {
        // r-1 is the largest valid scalar; exercises all windows being nonzero.
        let c = <Bls12 as GpuCurve>::bucket_width();
        let s = -Scalar::ONE; // r - 1
        let scalars = vec![s];
        let bd = compute_bucket_sorting::<Bls12>(&scalars);
        assert_bucket_data_invariants(&bd, 1, c);
        assert_bucket_data_covers_all_windows(&bd, &scalars, c);
    }

    #[test]
    fn bucket_sorting_powers_of_two() {
        // Powers of 2 exercise carry propagation across window boundaries.
        let c = <Bls12 as GpuCurve>::bucket_width();
        let scalars: Vec<Scalar> = (0..20).map(|i| Scalar::from(1u64 << i)).collect();
        let bd = compute_bucket_sorting::<Bls12>(&scalars);
        assert_bucket_data_invariants(&bd, 20, c);
        assert_bucket_data_covers_all_windows(&bd, &scalars, c);
    }

    /// Verify that the number of buckets per window is bounded by 2^(c-1).
    #[test]
    fn bucket_count_halved_vs_unsigned() {
        let c = <Bls12 as GpuCurve>::bucket_width();
        let half = (1u32 << (c - 1)) as usize;
        let scalars: Vec<Scalar> = (0..5000).map(|_| Scalar::random(OsRng)).collect();
        let bd = compute_bucket_sorting::<Bls12>(&scalars);

        for w in 0..bd.num_windows as usize {
            let count = bd.window_counts[w] as usize;
            assert!(
                count <= half,
                "window {w} has {count} active buckets, max is {half}"
            );
        }
    }

    /// Verify signed-digit decomposition is consistent with unsigned for reconstruction.
    /// The signed and unsigned decompositions must yield the same scalar.
    #[test]
    fn signed_vs_unsigned_decomposition_same_scalar() {
        let c = <Bls12 as GpuCurve>::bucket_width();
        let base = Scalar::from(1u64 << c);

        for _ in 0..50 {
            let s = Scalar::random(OsRng);
            let unsigned = <Bls12 as GpuCurve>::scalar_to_windows(&s, c);
            let signed = <Bls12 as GpuCurve>::scalar_to_signed_windows(&s, c);

            // Reconstruct from unsigned
            let mut val_u = Scalar::ZERO;
            let mut pow = Scalar::ONE;
            for &w in &unsigned {
                val_u += Scalar::from(w as u64) * pow;
                pow *= base;
            }

            // Reconstruct from signed
            let mut val_s = Scalar::ZERO;
            let mut pow = Scalar::ONE;
            for &(abs, neg) in &signed {
                let term = Scalar::from(abs as u64) * pow;
                if neg {
                    val_s -= term;
                } else {
                    val_s += term;
                }
                pow *= base;
            }

            assert_eq!(val_u, s, "unsigned reconstruction mismatch");
            assert_eq!(val_s, s, "signed reconstruction mismatch");
        }
    }

    /// Verify no window has abs == 2^(c-1) AND neg==false simultaneously
    /// (values at the boundary should always be negative to maintain the
    /// invariant that bucket values are in [1, 2^(c-1)]).
    #[test]
    fn signed_windows_boundary_invariant() {
        let c = <Bls12 as GpuCurve>::bucket_width();
        let half = 1u32 << (c - 1);

        // Specifically test scalars near window boundaries.
        let boundary_scalars: Vec<Scalar> = (0..c)
            .flat_map(|w| {
                let shift = (w * c) as u64;
                if shift < 64 {
                    vec![
                        Scalar::from((half as u64 - 1) << shift),
                        Scalar::from((half as u64) << shift),
                        Scalar::from((half as u64 + 1) << shift),
                    ]
                } else {
                    vec![]
                }
            })
            .collect();

        let all_scalars: Vec<Scalar> = boundary_scalars
            .into_iter()
            .chain((0..100).map(|_| Scalar::random(OsRng)))
            .collect();

        for s in &all_scalars {
            let signed = <Bls12 as GpuCurve>::scalar_to_signed_windows(s, c);
            for (w, &(abs, neg)) in signed.iter().enumerate() {
                assert!(
                    abs <= half,
                    "window {w}: abs={abs} exceeds half={half}"
                );
                // If abs == half, it must be negative (the boundary case)
                if abs == half {
                    assert!(
                        neg,
                        "window {w}: abs==2^(c-1)={half} but not negated — this means \
                         the bucket value would be out of the valid [1, 2^(c-1)] range \
                         for a positive digit"
                    );
                }
            }
        }
    }
}
