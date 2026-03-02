use super::*;
use blstrs::Bls12;
use ff::Field;
use rand_core::OsRng;

type Scalar = <Bls12 as GpuCurve>::Scalar;

const SIGN_BIT_MASK: u32 = 1 << 31;
const INDEX_MASK: u32 = !SIGN_BIT_MASK;

/// Verify all structural invariants of BucketData (with sub-bucket chunking).
fn assert_bucket_data_invariants(bd: &BucketData, n: usize, c: usize) {
    let half = 1u32 << (c - 1);

    // Parallel array lengths match num_dispatched (sub-bucket count)
    assert_eq!(bd.bucket_pointers.len(), bd.num_dispatched as usize);
    assert_eq!(bd.bucket_sizes.len(), bd.num_dispatched as usize);
    assert_eq!(bd.bucket_values.len(), bd.num_dispatched as usize);
    assert_eq!(bd.window_starts.len(), bd.num_windows as usize);
    assert_eq!(bd.window_counts.len(), bd.num_windows as usize);

    // Reduce arrays match num_active_buckets (original bucket count)
    assert_eq!(bd.reduce_starts.len(), bd.num_active_buckets as usize);
    assert_eq!(bd.reduce_counts.len(), bd.num_active_buckets as usize);
    assert!(bd.num_dispatched >= bd.num_active_buckets);

    // Sum of window_counts == num_dispatched
    let total_dispatched: u32 = bd.window_counts.iter().sum();
    assert_eq!(total_dispatched, bd.num_dispatched);

    // Sum of reduce_counts == num_dispatched
    let total_reduce: u32 = bd.reduce_counts.iter().sum();
    assert_eq!(total_reduce, bd.num_dispatched);

    // Sum of bucket_sizes == base_indices.len()
    let total_entries: u32 = bd.bucket_sizes.iter().sum();
    assert_eq!(total_entries as usize, bd.base_indices.len());

    // Every sub-bucket is non-empty, within curve chunk size, and has valid pointers
    for i in 0..bd.num_dispatched as usize {
        assert!(bd.bucket_sizes[i] > 0, "empty sub-bucket at index {i}");
        assert!(
            bd.bucket_sizes[i] <= <Bls12 as GpuCurve>::MSM_MAX_CHUNK_SIZE,
            "sub-bucket {i} has size {} > MSM_MAX_CHUNK_SIZE={}",
            bd.bucket_sizes[i],
            <Bls12 as GpuCurve>::MSM_MAX_CHUNK_SIZE
        );
        let ptr = bd.bucket_pointers[i] as usize;
        let end = ptr + bd.bucket_sizes[i] as usize;
        assert!(
            end <= bd.base_indices.len(),
            "sub-bucket {i} overflows base_indices"
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

    // Within each window, bucket values are in non-decreasing order
    // (sub-buckets of the same parent share the same value)
    for w in 0..bd.num_windows as usize {
        let start = bd.window_starts[w] as usize;
        let count = bd.window_counts[w] as usize;
        for j in 1..count {
            assert!(
                bd.bucket_values[start + j] >= bd.bucket_values[start + j - 1],
                "bucket values not sorted in window {w}"
            );
        }
    }

    // Reduce starts/counts are consistent: each original bucket's sub-buckets
    // are contiguous and cover exactly the right number of dispatched entries
    for j in 0..bd.num_active_buckets as usize {
        let start = bd.reduce_starts[j] as usize;
        let count = bd.reduce_counts[j] as usize;
        assert!(count >= 1, "original bucket {j} has 0 sub-buckets");
        assert!(
            start + count <= bd.num_dispatched as usize,
            "reduce range overflows for bucket {j}"
        );
        // All sub-buckets in this range must share the same bucket_value
        let val = bd.bucket_values[start];
        for k in 1..count {
            assert_eq!(
                bd.bucket_values[start + k],
                val,
                "sub-buckets of bucket {j} have mismatched values"
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
                assert!(prev.is_none(), "scalar {idx} appears twice in window {w}");
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

/// Verify that the number of *original* buckets per window is bounded by 2^(c-1).
/// With sub-bucket chunking, num_dispatched sub-buckets per window can exceed this
/// but the distinct bucket values per window must not.
#[test]
fn bucket_count_halved_vs_unsigned() {
    let c = <Bls12 as GpuCurve>::bucket_width();
    let half = (1u32 << (c - 1)) as usize;
    let scalars: Vec<Scalar> = (0..5000).map(|_| Scalar::random(OsRng)).collect();
    let bd = compute_bucket_sorting::<Bls12>(&scalars);

    for w in 0..bd.num_windows as usize {
        let start = bd.window_starts[w] as usize;
        let count = bd.window_counts[w] as usize;
        // Count distinct bucket values (original buckets) in this window.
        let mut seen = std::collections::HashSet::new();
        for j in 0..count {
            seen.insert(bd.bucket_values[start + j]);
        }
        assert!(
            seen.len() <= half,
            "window {w} has {} distinct bucket values, max is {half}",
            seen.len()
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

/// GLV bucket sorting: structural invariants hold for random scalars.
#[test]
fn glv_bucket_sorting_structural_invariants() {
    use crate::glv::bls12_381 as glv;
    use blstrs::G1Affine;
    use group::prime::PrimeCurveAffine;

    let c = 15usize;
    let g = G1Affine::generator();

    for &n in &[1, 2, 10, 50] {
        let scalars: Vec<Scalar> = (0..n).map(|_| Scalar::random(OsRng)).collect();

        let bases_bytes: Vec<u8> = (0..n)
            .flat_map(|_| <Bls12 as GpuCurve>::serialize_g1(&g))
            .collect();
        let phi_bases_bytes: Vec<u8> = (0..n)
            .flat_map(|_| {
                let phi = glv::endomorphism_g1(&g);
                <Bls12 as GpuCurve>::serialize_g1(&phi)
            })
            .collect();

        let (combined, bd) =
            compute_glv_bucket_sorting::<Bls12>(&scalars, &bases_bytes, &phi_bases_bytes, c);

        // Combined bases should have 2*n points of G1_GPU_BYTES bytes each
        assert_eq!(combined.len(), n * 2 * <Bls12 as GpuCurve>::G1_GPU_BYTES);

        // Window/sub-bucket parallel array lengths
        assert_eq!(bd.bucket_pointers.len(), bd.num_dispatched as usize);
        assert_eq!(bd.bucket_sizes.len(), bd.num_dispatched as usize);
        assert_eq!(bd.bucket_values.len(), bd.num_dispatched as usize);
        assert_eq!(bd.window_starts.len(), bd.num_windows as usize);
        assert_eq!(bd.window_counts.len(), bd.num_windows as usize);

        // Sum invariants
        let total_dispatched: u32 = bd.window_counts.iter().sum();
        assert_eq!(total_dispatched, bd.num_dispatched);
        let total_entries: u32 = bd.bucket_sizes.iter().sum();
        assert_eq!(total_entries as usize, bd.base_indices.len());

        // Bucket values in [1, 2^(c-1)] (signed-digit decomposition)
        for &v in &bd.bucket_values {
            assert!(v >= 1);
            assert!(v <= (1 << (c - 1)));
        }

        // All base indices (after masking sign bit) must be < 2*n
        for &idx in &bd.base_indices {
            let raw_idx = (idx & !<Bls12 as GpuCurve>::MSM_INDEX_SIGN_BIT) as usize;
            assert!(
                raw_idx < 2 * n,
                "GLV index {raw_idx} out of range for 2*n={}",
                2 * n
            );
        }
    }
}

/// GLV bucket sorting with all-zero scalars produces no buckets.
#[test]
fn glv_bucket_sorting_all_zero_scalars() {
    use crate::glv::bls12_381 as glv;
    use blstrs::G1Affine;
    use group::prime::PrimeCurveAffine;

    let c = 15usize;
    let g = G1Affine::generator();
    let n = 5;

    let scalars = vec![Scalar::ZERO; n];
    let bases_bytes: Vec<u8> = (0..n)
        .flat_map(|_| <Bls12 as GpuCurve>::serialize_g1(&g))
        .collect();
    let phi_bases_bytes: Vec<u8> = (0..n)
        .flat_map(|_| {
            let phi = glv::endomorphism_g1(&g);
            <Bls12 as GpuCurve>::serialize_g1(&phi)
        })
        .collect();

    let (combined, bd) =
        compute_glv_bucket_sorting::<Bls12>(&scalars, &bases_bytes, &phi_bases_bytes, c);

    assert_eq!(combined.len(), n * 2 * <Bls12 as GpuCurve>::G1_GPU_BYTES);
    assert_eq!(bd.num_active_buckets, 0);
    assert_eq!(bd.base_indices.len(), 0);
}

/// GLV bucket sorting: every scalar's windows land in the correct bucket.
#[test]
fn glv_bucket_sorting_window_correctness() {
    use crate::glv::bls12_381 as glv;
    use blstrs::G1Affine;
    use group::prime::PrimeCurveAffine;

    let c = 15usize;
    let g = G1Affine::generator();
    let n = 20;

    let scalars: Vec<Scalar> = (0..n).map(|_| Scalar::random(OsRng)).collect();
    let bases_bytes: Vec<u8> = (0..n)
        .flat_map(|_| <Bls12 as GpuCurve>::serialize_g1(&g))
        .collect();
    let phi_bases_bytes: Vec<u8> = (0..n)
        .flat_map(|_| {
            let phi = glv::endomorphism_g1(&g);
            <Bls12 as GpuCurve>::serialize_g1(&phi)
        })
        .collect();

    let (_combined, bd) =
        compute_glv_bucket_sorting::<Bls12>(&scalars, &bases_bytes, &phi_bases_bytes, c);

    // Recompute the expected signed-digit windows
    let mut expected_windows: Vec<Vec<(u32, bool)>> = Vec::new();
    for s in &scalars {
        let (k1, _k1_neg, k2, _k2_neg) = glv::glv_decompose(s);
        expected_windows.push(glv::u128_to_signed_windows(k1, c));
        expected_windows.push(glv::u128_to_signed_windows(k2, c));
    }

    // Verify each window: build map from (raw_index with sign) -> bucket_value
    for w in 0..bd.num_windows as usize {
        let w_start = bd.window_starts[w] as usize;
        let w_count = bd.window_counts[w] as usize;

        // Map from raw base_indices entry (with SIGN_BIT) to bucket value
        let mut found: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for b in 0..w_count {
            let bucket_idx = w_start + b;
            let val = bd.bucket_values[bucket_idx];
            let ptr = bd.bucket_pointers[bucket_idx] as usize;
            let size = bd.bucket_sizes[bucket_idx] as usize;
            for k in 0..size {
                let raw = bd.base_indices[ptr + k];
                found.insert(raw, val);
            }
        }

        // Every expected non-zero window must appear in found
        for (i, windows) in expected_windows.iter().enumerate() {
            if w < windows.len() {
                let (abs_val, is_neg) = windows[w];
                if abs_val != 0 {
                    let expected_key = if is_neg {
                        i as u32 | <Bls12 as GpuCurve>::MSM_INDEX_SIGN_BIT
                    } else {
                        i as u32
                    };
                    assert_eq!(
                        found.get(&expected_key),
                        Some(&abs_val),
                        "point {i} window {w}: expected bucket {abs_val} (neg={is_neg}), got {:?}",
                        found.get(&expected_key)
                    );
                }
            }
        }
    }
}

/// Bucket pointers form contiguous, non-overlapping spans in base_indices.
#[test]
fn bucket_pointers_are_contiguous() {
    let scalars: Vec<Scalar> = (0..500).map(|_| Scalar::random(OsRng)).collect();
    let bd = compute_bucket_sorting::<Bls12>(&scalars);

    if bd.num_dispatched == 0 {
        return;
    }

    // Verify sub-bucket spans are consecutive and cover base_indices exactly.
    let mut expected_ptr = 0u32;
    for i in 0..bd.num_dispatched as usize {
        assert_eq!(
            bd.bucket_pointers[i], expected_ptr,
            "sub-bucket {i} pointer gap: expected {expected_ptr}, got {}",
            bd.bucket_pointers[i]
        );
        expected_ptr += bd.bucket_sizes[i];
    }
    assert_eq!(expected_ptr as usize, bd.base_indices.len());
}

/// Each scalar appears at most once per window in the bucket data.
#[test]
fn no_duplicate_scalars_in_window() {
    let scalars: Vec<Scalar> = (0..200).map(|_| Scalar::random(OsRng)).collect();
    let bd = compute_bucket_sorting::<Bls12>(&scalars);

    for w in 0..bd.num_windows as usize {
        let w_start = bd.window_starts[w] as usize;
        let w_count = bd.window_counts[w] as usize;
        let mut seen = std::collections::HashSet::new();

        for b in 0..w_count {
            let bucket_idx = w_start + b;
            let ptr = bd.bucket_pointers[bucket_idx] as usize;
            let size = bd.bucket_sizes[bucket_idx] as usize;
            for k in 0..size {
                let idx = bd.base_indices[ptr + k] & INDEX_MASK;
                assert!(
                    seen.insert(idx),
                    "scalar {idx} appears more than once in window {w}"
                );
            }
        }
    }
}

/// With random scalars (no chunking), bucket values within each window have
/// no duplicates (each value is unique). With chunking, sub-buckets of the
/// same parent share a value, so duplicates are expected.
#[test]
fn bucket_values_unique_per_window_no_chunking() {
    // 300 random scalars at c=13 => ~1-2 points per bucket, no chunking
    let scalars: Vec<Scalar> = (0..300).map(|_| Scalar::random(OsRng)).collect();
    let bd = compute_bucket_sorting::<Bls12>(&scalars);
    assert!(
        !bd.has_chunks,
        "expected no chunking with 300 random scalars"
    );

    for w in 0..bd.num_windows as usize {
        let w_start = bd.window_starts[w] as usize;
        let w_count = bd.window_counts[w] as usize;
        let mut seen_vals = std::collections::HashSet::new();
        for b in 0..w_count {
            let val = bd.bucket_values[w_start + b];
            assert!(
                seen_vals.insert(val),
                "duplicate bucket value {val} in window {w}"
            );
        }
    }
}

/// Various small constant scalars produce expected window patterns.
#[test]
fn bucket_sorting_small_constants() {
    let c = <Bls12 as GpuCurve>::bucket_width();
    let half = 1u64 << (c - 1);

    // Scalar = 1 should land in bucket 1, window 0, positive
    let bd = compute_bucket_sorting::<Bls12>(&[Scalar::from(1u64)]);
    assert_eq!(bd.num_active_buckets, 1);
    assert_eq!(bd.bucket_values[0], 1);
    assert_eq!(bd.base_indices[0] & SIGN_BIT_MASK, 0);

    // Scalar = half-1 should be positive in window 0
    let bd = compute_bucket_sorting::<Bls12>(&[Scalar::from(half - 1)]);
    assert_eq!(bd.num_active_buckets, 1);
    assert_eq!(bd.bucket_values[0], (half - 1) as u32);
    assert_eq!(bd.base_indices[0] & SIGN_BIT_MASK, 0);

    // Scalar = half should be negative (boundary): abs=half, neg=true, carry=1
    let bd = compute_bucket_sorting::<Bls12>(&[Scalar::from(half)]);
    assert!(bd.num_active_buckets >= 1);
    // The first window should have the sign bit set
    let has_neg = bd.base_indices.iter().any(|&r| r & SIGN_BIT_MASK != 0);
    assert!(has_neg, "scalar=2^(c-1) should produce a negated entry");
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
            assert!(abs <= half, "window {w}: abs={abs} exceeds half={half}");
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

#[test]
fn optimal_glv_c_values() {
    // Tiny inputs get default
    assert_eq!(optimal_glv_c::<Bls12>(10), 13);
    assert_eq!(optimal_glv_c::<Bls12>(100), 13);

    // All values in valid range [10, 13]
    for n in [
        256, 500, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 500_000,
    ] {
        let c = optimal_glv_c::<Bls12>(n);
        assert!(c >= 10 && c <= 13, "c={c} out of range for n={n}");
    }

    // Larger inputs should prefer equal or larger c
    let c_small = optimal_glv_c::<Bls12>(1_000);
    let c_large = optimal_glv_c::<Bls12>(1_000_000);
    assert!(c_large >= c_small);
}
