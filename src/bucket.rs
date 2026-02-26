use crate::gpu::curve::GpuCurve;

pub struct BucketData<G: GpuCurve> {
    pub buckets: Vec<G::G1Affine>,
    pub indices: Vec<u32>,
    pub bucket_count: usize,
}

// TODO: optimize this
pub fn compute_bucket_sorting<G: GpuCurve>(
    bases: &[G::G1Affine],
    scalars: &[G::Scalar],
) -> BucketData<G> {
    let n = bases.len();
    assert_eq!(n, scalars.len());

    let c = G::bucket_width();
    let scalar_bits = <G::Scalar as ff::PrimeField>::NUM_BITS as usize;
    let num_windows = scalar_bits.div_ceil(c);

    let mut window_buckets: Vec<Vec<usize>> = (0..num_windows).map(|_| Vec::new()).collect();

    for (i, scalar) in scalars.iter().enumerate() {
        let windows = G::scalar_to_windows(scalar, c);
        for (j, window) in windows.iter().enumerate() {
            if *window != 0 {
                window_buckets[j].push(i);
            }
        }
    }

    let mut indices = Vec::with_capacity(n);
    let mut bucket_idx = 0u32;

    for window_bucket in window_buckets.iter().take(num_windows) {
        for _ in 0..window_bucket.len() {
            indices.push(bucket_idx);
            bucket_idx += 1;
        }
    }

    let bucket_count = indices.len();
    let mut buckets = Vec::with_capacity(bucket_count);

    for window_bucket in window_buckets.iter().take(num_windows) {
        for &base_idx in window_bucket {
            buckets.push(bases[base_idx].clone());
        }
    }

    BucketData {
        buckets,
        indices,
        bucket_count,
    }
}
