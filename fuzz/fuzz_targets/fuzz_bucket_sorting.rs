#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use blstrs::Scalar;
use ff::PrimeField;
use webgpu_groth16::bucket::compute_bucket_sorting_with_width;

#[derive(Arbitrary, Debug)]
struct Input {
    /// Raw scalar bytes. Each 32-byte chunk becomes one scalar.
    /// Using a fixed-size array to keep input manageable for the fuzzer.
    scalar_bytes: Vec<[u8; 32]>,
    /// Bucket width, will be clamped to [8, 13]
    c_raw: u8,
}

fuzz_target!(|input: Input| {
    let c = (input.c_raw % 6) as usize + 8; // c ∈ [8, 13]

    // Parse scalars, skipping invalid ones
    let scalars: Vec<Scalar> = input
        .scalar_bytes
        .iter()
        .filter_map(|bytes| Scalar::from_repr_vartime(*bytes))
        .collect();

    // Need at least 1 scalar, cap at 128 to keep fuzzing fast
    if scalars.is_empty() || scalars.len() > 128 {
        return;
    }

    let data = compute_bucket_sorting_with_width::<blstrs::Bls12>(&scalars, c);
    let half = 1u32 << (c - 1);

    // Property 1: total points across all buckets == base_indices.len()
    let total_points: u32 = data.bucket_sizes.iter().sum();
    assert_eq!(
        total_points,
        data.base_indices.len() as u32,
        "total bucket sizes should equal base_indices length"
    );

    // Property 2: all bucket values in [1, 2^(c-1)]
    for (i, &val) in data.bucket_values.iter().enumerate() {
        assert!(
            val >= 1 && val <= half,
            "bucket {i}: value {val} not in [1, {half}]"
        );
    }

    // Property 3: bucket pointers are contiguous
    for i in 0..data.bucket_pointers.len().saturating_sub(1) {
        let expected_next = data.bucket_pointers[i] + data.bucket_sizes[i];
        assert_eq!(
            data.bucket_pointers[i + 1], expected_next,
            "bucket pointers not contiguous at index {i}"
        );
    }

    // Property 4: window starts + counts are consistent
    assert_eq!(data.window_starts.len(), data.num_windows as usize);
    assert_eq!(data.window_counts.len(), data.num_windows as usize);

    let total_dispatched: u32 = data.window_counts.iter().sum();
    assert_eq!(
        total_dispatched, data.num_dispatched,
        "sum of window_counts should equal num_dispatched"
    );

    // Property 5: all base indices (masked) are valid point indices
    let n = scalars.len() as u32;
    let sign_bit = 1u32 << 31;
    for (i, &idx) in data.base_indices.iter().enumerate() {
        let point_idx = idx & !sign_bit;
        assert!(
            point_idx < n,
            "base_indices[{i}] = {point_idx} (masked) >= n={n}"
        );
    }

    // Property 6: sub-bucket sizes are bounded by MAX_CHUNK_SIZE (64)
    for (i, &size) in data.bucket_sizes.iter().enumerate() {
        assert!(
            size <= 64,
            "bucket_sizes[{i}] = {size} exceeds MAX_CHUNK_SIZE=64"
        );
    }
});
