use ff::PrimeField;

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

pub fn compute_bucket_sorting<G: GpuCurve>(scalars: &[G::Scalar]) -> BucketData {
    compute_bucket_sorting_with_width::<G>(scalars, G::bucket_width())
}

pub fn compute_bucket_sorting_with_width<G: GpuCurve>(
    scalars: &[G::Scalar],
    c: usize,
) -> BucketData {
    let c = c;
    let scalar_bits = <G::Scalar as PrimeField>::NUM_BITS as usize;
    let num_windows = scalar_bits.div_ceil(c);

    let mut base_indices = Vec::new();
    let mut bucket_pointers = Vec::new();
    let mut bucket_sizes = Vec::new();
    let mut bucket_values = Vec::new();
    let mut window_starts = Vec::new();
    let mut window_counts = Vec::new();

    for w in 0..num_windows {
        // Fast 2D array for sparse bucketing. c=15 means 32768 max value.
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); 1 << c];

        for (i, scalar) in scalars.iter().enumerate() {
            let windows = G::scalar_to_windows(scalar, c);
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
