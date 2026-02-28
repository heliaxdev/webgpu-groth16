// TODO: implement a form of efficient batching of GPU operations.
// loading data in and out of the GPU is slow. we want to batch as many ops as possible,
// without using a stupid amount of VRAM (perhaps we need to check
// how much VRAM the host has).

use anyhow::Result;
use ff::{Field, PrimeField};
use rand_core::RngCore;

use crate::bellman;
use crate::bucket::{BucketData, compute_bucket_sorting, compute_bucket_sorting_with_width};
use crate::gpu::{GpuContext, curve::GpuCurve};

struct GpuConstraintSystem<G: GpuCurve> {
    inputs: Vec<G::Scalar>,
    aux: Vec<G::Scalar>,
    a_aux_density: Vec<bool>,
    b_input_density: Vec<bool>,
    b_aux_density: Vec<bool>,
    a_lcs: Vec<Vec<(bellman::Variable, G::Scalar)>>,
    b_lcs: Vec<Vec<(bellman::Variable, G::Scalar)>>,
    c_lcs: Vec<Vec<(bellman::Variable, G::Scalar)>>,
    _marker: std::marker::PhantomData<G>,
}

impl<G: GpuCurve> Default for GpuConstraintSystem<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G: GpuCurve> GpuConstraintSystem<G> {
    fn new() -> Self {
        GpuConstraintSystem {
            inputs: vec![G::Scalar::ONE],
            aux: Vec::new(),
            a_aux_density: Vec::new(),
            b_input_density: vec![false],
            b_aux_density: Vec::new(),
            a_lcs: Vec::new(),
            b_lcs: Vec::new(),
            c_lcs: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<G: GpuCurve + Send> bellman::ConstraintSystem<G::Scalar> for GpuConstraintSystem<G> {
    type Root = Self;
    fn alloc<F, A, AR>(
        &mut self,
        _annotation: A,
        f: F,
    ) -> Result<bellman::Variable, bellman::SynthesisError>
    where
        F: FnOnce() -> Result<G::Scalar, bellman::SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let value = f()?;
        self.aux.push(value);
        self.a_aux_density.push(false);
        self.b_aux_density.push(false);
        Ok(bellman::Variable::new_unchecked(bellman::Index::Aux(
            self.aux.len() - 1,
        )))
    }
    fn alloc_input<F, A, AR>(
        &mut self,
        _annotation: A,
        f: F,
    ) -> Result<bellman::Variable, bellman::SynthesisError>
    where
        F: FnOnce() -> Result<G::Scalar, bellman::SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let value = f()?;
        self.inputs.push(value);
        self.b_input_density.push(false);
        Ok(bellman::Variable::new_unchecked(bellman::Index::Input(
            self.inputs.len() - 1,
        )))
    }
    fn enforce<A, AR, LA, LB, LC>(&mut self, _annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(bellman::LinearCombination<G::Scalar>) -> bellman::LinearCombination<G::Scalar>,
        LB: FnOnce(bellman::LinearCombination<G::Scalar>) -> bellman::LinearCombination<G::Scalar>,
        LC: FnOnce(bellman::LinearCombination<G::Scalar>) -> bellman::LinearCombination<G::Scalar>,
    {
        let a_lc = a(bellman::LinearCombination::zero());
        let b_lc = b(bellman::LinearCombination::zero());
        let c_lc = c(bellman::LinearCombination::zero());

        let a_vec = lc_to_vec(a_lc);
        for (var, coeff) in &a_vec {
            if *coeff == G::Scalar::ZERO {
                continue;
            }
            if let bellman::Index::Aux(i) = var.get_unchecked() {
                self.a_aux_density[i] = true;
            }
        }

        let b_vec = lc_to_vec(b_lc);
        for (var, coeff) in &b_vec {
            if *coeff == G::Scalar::ZERO {
                continue;
            }
            match var.get_unchecked() {
                bellman::Index::Input(i) => self.b_input_density[i] = true,
                bellman::Index::Aux(i) => self.b_aux_density[i] = true,
            }
        }

        self.a_lcs.push(a_vec);
        self.b_lcs.push(b_vec);
        self.c_lcs.push(lc_to_vec(c_lc));
    }
    fn push_namespace<NR, N>(&mut self, _name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }
    fn pop_namespace(&mut self) {}
    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}

fn lc_to_vec<S: PrimeField>(lc: bellman::LinearCombination<S>) -> Vec<(bellman::Variable, S)> {
    #[cfg(feature = "bellman-provider-bellman")]
    let lc_iter = lc.as_ref().iter();

    #[cfg(feature = "bellman-provider-nam-bellperson")]
    let lc_iter = lc.iter();

    lc_iter.map(|(var, coeff)| (var, *coeff)).collect()
}

/// Pre-serialized proving key bases for GPU. Avoids re-serialization per proof.
pub struct PreparedProvingKey<G: GpuCurve> {
    pub a_bytes: Vec<u8>,
    pub b_g1_bytes: Vec<u8>,
    pub l_bytes: Vec<u8>,
    pub h_bytes: Vec<u8>,
    pub b_g2_bytes: Vec<u8>,
    _marker: std::marker::PhantomData<G>,
}

fn serialize_g1_bases<G: GpuCurve>(bases: &[G::G1Affine]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(bases.len() * 144);
    for base in bases {
        bytes.extend_from_slice(&G::serialize_g1(base));
    }
    bytes
}

fn serialize_g2_bases<G: GpuCurve>(bases: &[G::G2Affine]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(bases.len() * 288);
    for base in bases {
        bytes.extend_from_slice(&G::serialize_g2(base));
    }
    bytes
}

pub fn prepare_proving_key<E, G>(pk: &bellman::groth16::Parameters<E>) -> PreparedProvingKey<G>
where
    E: pairing::MultiMillerLoop,
    G: GpuCurve<
        Engine = E,
        Scalar = E::Fr,
        G1 = E::G1,
        G2 = E::G2,
        G1Affine = E::G1Affine,
        G2Affine = E::G2Affine,
    >,
{
    PreparedProvingKey {
        a_bytes: serialize_g1_bases::<G>(&pk.a),
        b_g1_bytes: serialize_g1_bases::<G>(&pk.b_g1),
        l_bytes: serialize_g1_bases::<G>(&pk.l),
        h_bytes: serialize_g1_bases::<G>(&pk.h),
        b_g2_bytes: serialize_g2_bases::<G>(&pk.b_g2),
        _marker: std::marker::PhantomData,
    }
}

fn marshal_scalars<G: GpuCurve>(scalars: &[G::Scalar]) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(scalars.len() * 32);
    for s in scalars {
        buffer.extend_from_slice(&G::serialize_scalar(s));
    }
    buffer
}

fn dense_assignment_from_masks<S: PrimeField>(
    inputs: &[S],
    aux: &[S],
    input_mask: &[bool],
    aux_mask: &[bool],
) -> Vec<S> {
    let mut out = Vec::new();
    for (i, s) in inputs.iter().enumerate() {
        if i < input_mask.len() && input_mask[i] {
            out.push(*s);
        }
    }
    for (i, s) in aux.iter().enumerate() {
        if i < aux_mask.len() && aux_mask[i] {
            out.push(*s);
        }
    }
    out
}

pub async fn gpu_msm_g1<G: GpuCurve>(
    gpu: &GpuContext<G>,
    bases: &[G::G1Affine],
    scalars: &[G::Scalar],
) -> Result<G::G1> {
    #[cfg(feature = "timing")]
    let t_start = std::time::Instant::now();

    let bd: BucketData = compute_bucket_sorting::<G>(scalars);
    if bd.num_active_buckets == 0 {
        return Ok(G::g1_identity());
    }

    #[cfg(feature = "timing")]
    let t_bucket = std::time::Instant::now();

    let mut bases_bytes = Vec::with_capacity(bases.len() * 144);
    for base in bases {
        bases_bytes.extend_from_slice(&G::serialize_g1(base));
    }

    let bases_buf = gpu.create_storage_buffer("Bases", &bases_bytes);
    let indices_buf = gpu.create_storage_buffer("Indices", bytemuck::cast_slice(&bd.base_indices));
    let ptrs_buf = gpu.create_storage_buffer("Ptrs", bytemuck::cast_slice(&bd.bucket_pointers));
    let sizes_buf = gpu.create_storage_buffer("Sizes", bytemuck::cast_slice(&bd.bucket_sizes));
    let vals_buf = gpu.create_storage_buffer("Vals", bytemuck::cast_slice(&bd.bucket_values));
    let w_starts_buf =
        gpu.create_storage_buffer("WStarts", bytemuck::cast_slice(&bd.window_starts));
    let w_counts_buf =
        gpu.create_storage_buffer("WCounts", bytemuck::cast_slice(&bd.window_counts));

    let agg_buf = gpu.create_empty_buffer("Agg", (bd.num_active_buckets * 144) as u64);
    let sums_buf = gpu.create_empty_buffer("Sums", (bd.num_windows * 144) as u64);

    #[cfg(feature = "timing")]
    let t_upload = std::time::Instant::now();

    gpu.execute_msm(
        false,
        &bases_buf,
        &indices_buf,
        &ptrs_buf,
        &sizes_buf,
        &agg_buf,
        &vals_buf,
        &w_starts_buf,
        &w_counts_buf,
        &sums_buf,
        bd.num_active_buckets,
        bd.num_windows,
    );

    #[cfg(feature = "timing")]
    let t_dispatch = std::time::Instant::now();

    let result_bytes = gpu
        .read_buffer(&sums_buf, (bd.num_windows * 144) as u64)
        .await?;

    #[cfg(feature = "timing")]
    let t_read = std::time::Instant::now();

    let result = fold_window_sums_g1::<G>(&result_bytes, bd.num_windows, G::bucket_width())?;

    #[cfg(feature = "timing")]
    {
        let t_fold = std::time::Instant::now();
        eprintln!(
            "[timing] msm_g1 n={}: bucket_sort={:?} upload={:?} dispatch+gpu={:?} readback={:?} fold={:?} total={:?}",
            scalars.len(),
            t_bucket.duration_since(t_start),
            t_upload.duration_since(t_bucket),
            t_dispatch.duration_since(t_upload),
            t_read.duration_since(t_dispatch),
            t_fold.duration_since(t_read),
            t_fold.duration_since(t_start),
        );
    }

    Ok(result)
}

async fn gpu_msm_g2<G: GpuCurve>(
    gpu: &GpuContext<G>,
    bases: &[G::G2Affine],
    scalars: &[G::Scalar],
) -> Result<G::G2> {
    let bd: BucketData = compute_bucket_sorting_with_width::<G>(scalars, G::g2_bucket_width());
    if bd.num_active_buckets == 0 {
        return Ok(G::g2_identity());
    }

    let mut bases_bytes = Vec::with_capacity(bases.len() * 288);
    for base in bases {
        bases_bytes.extend_from_slice(&G::serialize_g2(base));
    }

    let bases_buf = gpu.create_storage_buffer("BasesG2", &bases_bytes);
    let indices_buf = gpu.create_storage_buffer("Indices", bytemuck::cast_slice(&bd.base_indices));
    let ptrs_buf = gpu.create_storage_buffer("Ptrs", bytemuck::cast_slice(&bd.bucket_pointers));
    let sizes_buf = gpu.create_storage_buffer("Sizes", bytemuck::cast_slice(&bd.bucket_sizes));
    let vals_buf = gpu.create_storage_buffer("Vals", bytemuck::cast_slice(&bd.bucket_values));
    let w_starts_buf =
        gpu.create_storage_buffer("WStarts", bytemuck::cast_slice(&bd.window_starts));
    let w_counts_buf =
        gpu.create_storage_buffer("WCounts", bytemuck::cast_slice(&bd.window_counts));

    let agg_buf = gpu.create_empty_buffer("AggG2", (bd.num_active_buckets * 288) as u64);
    let sums_buf = gpu.create_empty_buffer("SumsG2", (bd.num_windows * 288) as u64);

    gpu.execute_msm(
        true,
        &bases_buf,
        &indices_buf,
        &ptrs_buf,
        &sizes_buf,
        &agg_buf,
        &vals_buf,
        &w_starts_buf,
        &w_counts_buf,
        &sums_buf,
        bd.num_active_buckets,
        bd.num_windows,
    );

    let result_bytes = gpu
        .read_buffer(&sums_buf, (bd.num_windows * 288) as u64)
        .await?;
    fold_window_sums_g2::<G>(&result_bytes, bd.num_windows, G::g2_bucket_width())
}

fn fold_window_sums_g1<G: GpuCurve>(
    result_bytes: &[u8],
    num_windows: u32,
    bucket_width: usize,
) -> Result<G::G1> {
    let mut result = G::g1_identity();
    for (i, chunk) in result_bytes.chunks_exact(144).enumerate().rev() {
        if i != (num_windows - 1) as usize {
            for _ in 0..bucket_width {
                result = G::add_g1_proj(&result, &result);
            }
        }
        let w_sum = G::deserialize_g1(chunk)?;
        result = G::add_g1_proj(&result, &w_sum);
    }
    Ok(result)
}

fn fold_window_sums_g2<G: GpuCurve>(
    result_bytes: &[u8],
    num_windows: u32,
    bucket_width: usize,
) -> Result<G::G2> {
    let mut result = G::g2_identity();
    for (i, chunk) in result_bytes.chunks_exact(288).enumerate().rev() {
        if i != (num_windows - 1) as usize {
            for _ in 0..bucket_width {
                result = G::add_g2_proj(&result, &result);
            }
        }
        let w_sum = G::deserialize_g2(chunk)?;
        result = G::add_g2_proj(&result, &w_sum);
    }
    Ok(result)
}

#[allow(clippy::type_complexity)]
pub async fn gpu_msm_batch<G: GpuCurve>(
    gpu: &GpuContext<G>,
    a_bases: &[G::G1Affine],
    a_scalars: &[G::Scalar],
    b1_bases: &[G::G1Affine],
    b_scalars: &[G::Scalar],
    l_bases: &[G::G1Affine],
    l_scalars: &[G::Scalar],
    h_bases: &[G::G1Affine],
    h_scalars: &[G::Scalar],
    b2_bases: &[G::G2Affine],
    b2_scalars: &[G::Scalar],
) -> Result<(G::G1, G::G1, G::G1, G::G1, G::G2)> {
    let a_bd = compute_bucket_sorting::<G>(a_scalars);
    let b_bd = compute_bucket_sorting::<G>(b_scalars);
    let l_bd = compute_bucket_sorting::<G>(l_scalars);
    let h_bd = compute_bucket_sorting::<G>(h_scalars);
    let b2_bd = compute_bucket_sorting_with_width::<G>(b2_scalars, G::g2_bucket_width());
    gpu_msm_batch_bytes::<G>(
        gpu,
        &serialize_g1_bases::<G>(a_bases),
        a_bd,
        &serialize_g1_bases::<G>(b1_bases),
        b_bd,
        &serialize_g1_bases::<G>(l_bases),
        l_bd,
        &serialize_g1_bases::<G>(h_bases),
        h_bd,
        &serialize_g2_bases::<G>(b2_bases),
        b2_bd,
    )
    .await
}

#[allow(clippy::type_complexity)]
async fn gpu_msm_batch_bytes<G: GpuCurve>(
    gpu: &GpuContext<G>,
    a_bytes: &[u8],
    a_bd: BucketData,
    b1_bytes: &[u8],
    b1_bd: BucketData,
    l_bytes: &[u8],
    l_bd: BucketData,
    h_bytes: &[u8],
    h_bd: BucketData,
    b2_bytes: &[u8],
    b2_bd: BucketData,
) -> Result<(G::G1, G::G1, G::G1, G::G1, G::G2)> {
    struct G1Pending {
        sums_buf: wgpu::Buffer,
        num_windows: u32,
    }
    struct G2Pending {
        sums_buf: wgpu::Buffer,
        num_windows: u32,
    }

    let enqueue_g1 =
        |name: &str, bases_bytes: &[u8], bd: BucketData| -> Result<Option<G1Pending>> {
            if bd.num_active_buckets == 0 {
                return Ok(None);
            }

            let bases_buf = gpu.create_storage_buffer(&format!("{name}_bases"), bases_bytes);
            let indices_buf = gpu.create_storage_buffer(
                &format!("{name}_indices"),
                bytemuck::cast_slice(&bd.base_indices),
            );
            let ptrs_buf = gpu.create_storage_buffer(
                &format!("{name}_ptrs"),
                bytemuck::cast_slice(&bd.bucket_pointers),
            );
            let sizes_buf = gpu.create_storage_buffer(
                &format!("{name}_sizes"),
                bytemuck::cast_slice(&bd.bucket_sizes),
            );
            let vals_buf = gpu.create_storage_buffer(
                &format!("{name}_vals"),
                bytemuck::cast_slice(&bd.bucket_values),
            );
            let w_starts_buf = gpu.create_storage_buffer(
                &format!("{name}_wstarts"),
                bytemuck::cast_slice(&bd.window_starts),
            );
            let w_counts_buf = gpu.create_storage_buffer(
                &format!("{name}_wcounts"),
                bytemuck::cast_slice(&bd.window_counts),
            );
            let agg_buf = gpu
                .create_empty_buffer(&format!("{name}_agg"), (bd.num_active_buckets * 144) as u64);
            let sums_buf =
                gpu.create_empty_buffer(&format!("{name}_sums"), (bd.num_windows * 144) as u64);

            gpu.execute_msm(
                false,
                &bases_buf,
                &indices_buf,
                &ptrs_buf,
                &sizes_buf,
                &agg_buf,
                &vals_buf,
                &w_starts_buf,
                &w_counts_buf,
                &sums_buf,
                bd.num_active_buckets,
                bd.num_windows,
            );

            Ok(Some(G1Pending {
                sums_buf,
                num_windows: bd.num_windows,
            }))
        };

    let enqueue_g2 = |bases_bytes: &[u8], bd: BucketData| -> Result<Option<G2Pending>> {
        if bd.num_active_buckets == 0 {
            return Ok(None);
        }

        let bases_buf = gpu.create_storage_buffer("b2_bases", bases_bytes);
        let indices_buf =
            gpu.create_storage_buffer("b2_indices", bytemuck::cast_slice(&bd.base_indices));
        let ptrs_buf =
            gpu.create_storage_buffer("b2_ptrs", bytemuck::cast_slice(&bd.bucket_pointers));
        let sizes_buf =
            gpu.create_storage_buffer("b2_sizes", bytemuck::cast_slice(&bd.bucket_sizes));
        let vals_buf =
            gpu.create_storage_buffer("b2_vals", bytemuck::cast_slice(&bd.bucket_values));
        let w_starts_buf =
            gpu.create_storage_buffer("b2_wstarts", bytemuck::cast_slice(&bd.window_starts));
        let w_counts_buf =
            gpu.create_storage_buffer("b2_wcounts", bytemuck::cast_slice(&bd.window_counts));
        let agg_buf = gpu.create_empty_buffer("b2_agg", (bd.num_active_buckets * 288) as u64);
        let sums_buf = gpu.create_empty_buffer("b2_sums", (bd.num_windows * 288) as u64);

        gpu.execute_msm(
            true,
            &bases_buf,
            &indices_buf,
            &ptrs_buf,
            &sizes_buf,
            &agg_buf,
            &vals_buf,
            &w_starts_buf,
            &w_counts_buf,
            &sums_buf,
            bd.num_active_buckets,
            bd.num_windows,
        );

        Ok(Some(G2Pending {
            sums_buf,
            num_windows: bd.num_windows,
        }))
    };

    let a_job = enqueue_g1("a", a_bytes, a_bd)?;
    let b1_job = enqueue_g1("b1", b1_bytes, b1_bd)?;
    let l_job = enqueue_g1("l", l_bytes, l_bd)?;
    let h_job = enqueue_g1("h", h_bytes, h_bd)?;
    let b2_job = enqueue_g2(b2_bytes, b2_bd)?;

    let mut read_targets: Vec<(&wgpu::Buffer, wgpu::BufferAddress)> = Vec::new();
    if let Some(job) = &a_job {
        read_targets.push((&job.sums_buf, (job.num_windows * 144) as u64));
    }
    if let Some(job) = &b1_job {
        read_targets.push((&job.sums_buf, (job.num_windows * 144) as u64));
    }
    if let Some(job) = &l_job {
        read_targets.push((&job.sums_buf, (job.num_windows * 144) as u64));
    }
    if let Some(job) = &h_job {
        read_targets.push((&job.sums_buf, (job.num_windows * 144) as u64));
    }
    if let Some(job) = &b2_job {
        read_targets.push((&job.sums_buf, (job.num_windows * 288) as u64));
    }

    let mut read_results = gpu.read_buffers_batch(&read_targets).await?.into_iter();

    let a = if let Some(job) = &a_job {
        fold_window_sums_g1::<G>(&read_results.next().unwrap(), job.num_windows, G::bucket_width())?
    } else {
        G::g1_identity()
    };
    let b1 = if let Some(job) = &b1_job {
        fold_window_sums_g1::<G>(&read_results.next().unwrap(), job.num_windows, G::bucket_width())?
    } else {
        G::g1_identity()
    };
    let l = if let Some(job) = &l_job {
        fold_window_sums_g1::<G>(&read_results.next().unwrap(), job.num_windows, G::bucket_width())?
    } else {
        G::g1_identity()
    };
    let h = if let Some(job) = &h_job {
        fold_window_sums_g1::<G>(&read_results.next().unwrap(), job.num_windows, G::bucket_width())?
    } else {
        G::g1_identity()
    };
    let b2 = if let Some(job) = &b2_job {
        fold_window_sums_g2::<G>(&read_results.next().unwrap(), job.num_windows, G::g2_bucket_width())?
    } else {
        G::g2_identity()
    };

    Ok((a, b1, l, h, b2))
}

fn eval_lc<S: PrimeField>(lc: &[(bellman::Variable, S)], inputs: &[S], aux: &[S]) -> S {
    let mut res = S::ZERO;
    for &(var, coeff) in lc {
        let val = match var.get_unchecked() {
            bellman::Index::Input(i) => inputs[i],
            bellman::Index::Aux(i) => aux[i],
        };
        let mut term = val;
        term.mul_assign(&coeff);
        res.add_assign(&term);
    }
    res
}

struct HPolyPending {
    h_buf: wgpu::Buffer,
    n: usize,
}

/// Submit the H polynomial pipeline to the GPU (non-blocking).
/// Returns a pending handle that can be read later with `read_h_poly_result`.
fn submit_h_poly<G: GpuCurve>(
    gpu: &GpuContext<G>,
    a_values: &[G::Scalar],
    b_values: &[G::Scalar],
    c_values: &[G::Scalar],
) -> Result<HPolyPending> {
    let n = a_values.len().next_power_of_two();

    // 1. CPU PRE-COMPUTES CONSTANT FACTORS
    let omega_n = G::root_of_unity(n);
    let omega_n_inv = omega_n.invert().unwrap();

    let n_inv = G::Scalar::from(n as u64).invert().unwrap();
    let coset_generator = G::Scalar::MULTIPLICATIVE_GENERATOR;
    let coset_inv = coset_generator.invert().unwrap();

    let mut inv_twiddles_n = vec![G::Scalar::ONE; n];
    let mut fwd_twiddles_n = vec![G::Scalar::ONE; n];
    let mut shifts = vec![G::Scalar::ONE; n];
    let mut inv_shifts = vec![G::Scalar::ONE; n];

    for i in 1..n {
        inv_twiddles_n[i] = inv_twiddles_n[i - 1] * omega_n_inv;
        fwd_twiddles_n[i] = fwd_twiddles_n[i - 1] * omega_n;
        shifts[i] = shifts[i - 1] * coset_generator;
        inv_shifts[i] = inv_shifts[i - 1] * coset_inv;
    }

    for i in 0..n {
        shifts[i] *= n_inv;
        inv_shifts[i] *= n_inv;
    }

    let g_to_n = coset_generator.pow([n as u64]);
    let z_inv = (g_to_n - G::Scalar::ONE).invert().unwrap();
    let z_invs = vec![z_inv];

    // 2. CPU ZERO-PADDING
    let mut a_coeffs = a_values.to_vec();
    a_coeffs.resize(n, G::Scalar::ZERO);
    let mut b_coeffs = b_values.to_vec();
    b_coeffs.resize(n, G::Scalar::ZERO);
    let mut c_coeffs = c_values.to_vec();
    c_coeffs.resize(n, G::Scalar::ZERO);

    // 3. UPLOAD ALL BUFFERS TO VRAM
    let a_buf = gpu.create_storage_buffer("A", &marshal_scalars::<G>(&a_coeffs));
    let b_buf = gpu.create_storage_buffer("B", &marshal_scalars::<G>(&b_coeffs));
    let c_buf = gpu.create_storage_buffer("C", &marshal_scalars::<G>(&c_coeffs));
    let h_buf = gpu.create_empty_buffer("H", (n * 32) as u64);

    let tw_inv_n_buf = gpu.create_storage_buffer("TwInvN", &marshal_scalars::<G>(&inv_twiddles_n));
    let tw_fwd_n_buf = gpu.create_storage_buffer("TwFwdN", &marshal_scalars::<G>(&fwd_twiddles_n));
    let shifts_buf = gpu.create_storage_buffer("Shifts", &marshal_scalars::<G>(&shifts));
    let inv_shifts_buf = gpu.create_storage_buffer("InvShifts", &marshal_scalars::<G>(&inv_shifts));
    let z_invs_buf = gpu.create_storage_buffer("ZInvs", &marshal_scalars::<G>(&z_invs));

    // 4. DISPATCH FULL H PIPELINE ON GPU USING A SINGLE COMMAND BUFFER
    gpu.execute_h_pipeline(
        &a_buf,
        &b_buf,
        &c_buf,
        &h_buf,
        &tw_inv_n_buf,
        &tw_fwd_n_buf,
        &shifts_buf,
        &inv_shifts_buf,
        &z_invs_buf,
        n as u32,
    );

    Ok(HPolyPending { h_buf, n })
}

/// Read the result of a previously submitted H polynomial pipeline.
async fn read_h_poly_result<G: GpuCurve>(
    gpu: &GpuContext<G>,
    pending: HPolyPending,
) -> Result<Vec<G::Scalar>> {
    let h_bytes = gpu
        .read_buffer(&pending.h_buf, (pending.n * 32) as wgpu::BufferAddress)
        .await?;

    let mut h_poly = vec![G::Scalar::ZERO; pending.n];
    for (i, chunk) in h_bytes.chunks_exact(32).enumerate() {
        h_poly[i] = G::deserialize_scalar(chunk)?;
    }

    Ok(h_poly)
}

pub async fn compute_h_poly<G: GpuCurve>(
    gpu: &GpuContext<G>,
    a_values: &[G::Scalar],
    b_values: &[G::Scalar],
    c_values: &[G::Scalar],
) -> Result<Vec<G::Scalar>> {
    let pending = submit_h_poly::<G>(gpu, a_values, b_values, c_values)?;
    read_h_poly_result::<G>(gpu, pending).await
}

async fn create_proof_with_fixed_randomness<E, G, C>(
    circuit: C,
    pk: &bellman::groth16::Parameters<E>,
    ppk: &PreparedProvingKey<G>,
    gpu: &GpuContext<G>,
    r: G::Scalar,
    s: G::Scalar,
) -> Result<bellman::groth16::Proof<E>>
where
    E: pairing::MultiMillerLoop,
    C: bellman::Circuit<G::Scalar>,
    G: GpuCurve<
            Engine = E,
            Scalar = E::Fr,
            G1 = E::G1,
            G2 = E::G2,
            G1Affine = E::G1Affine,
            G2Affine = E::G2Affine,
        > + Send,
{
    let mut cs = GpuConstraintSystem::<G>::new();
    circuit
        .synthesize(&mut cs)
        .map_err(|e| anyhow::anyhow!("circuit synthesis failed: {:?}", e))?;

    for i in 0..cs.inputs.len() {
        cs.a_lcs.push(vec![(
            bellman::Variable::new_unchecked(bellman::Index::Input(i)),
            G::Scalar::ONE,
        )]);
        cs.b_lcs.push(Vec::new());
        cs.c_lcs.push(Vec::new());
    }

    let num_constraints = cs.a_lcs.len();
    let n = num_constraints.next_power_of_two();

    let mut a_values = vec![G::Scalar::ZERO; n];
    let mut b_values = vec![G::Scalar::ZERO; n];
    let mut c_values = vec![G::Scalar::ZERO; n];

    for i in 0..num_constraints {
        a_values[i] = eval_lc(&cs.a_lcs[i], &cs.inputs, &cs.aux);
        b_values[i] = eval_lc(&cs.b_lcs[i], &cs.inputs, &cs.aux);
        c_values[i] = eval_lc(&cs.c_lcs[i], &cs.inputs, &cs.aux);
    }

    // Build assignments before H poly so we can pre-compute bucket data
    // while the GPU processes the H polynomial pipeline.
    let mut a_assignment = cs.inputs.clone();
    for (i, v) in cs.aux.iter().enumerate() {
        if cs.a_aux_density[i] {
            a_assignment.push(*v);
        }
    }
    let b_assignment =
        dense_assignment_from_masks(&cs.inputs, &cs.aux, &cs.b_input_density, &cs.b_aux_density);

    #[cfg(feature = "timing")]
    let t_h_start = std::time::Instant::now();

    // Submit H polynomial to GPU (non-blocking — GPU processes asynchronously)
    let h_pending = submit_h_poly::<G>(gpu, &a_values, &b_values, &c_values)?;

    // Pre-compute bucket data for non-H MSMs while GPU computes H
    let a_bd = compute_bucket_sorting::<G>(&a_assignment);
    let b1_bd = compute_bucket_sorting::<G>(&b_assignment);
    let l_bd = compute_bucket_sorting::<G>(&cs.aux);
    let b2_bd = compute_bucket_sorting_with_width::<G>(&b_assignment, G::g2_bucket_width());

    // Await H result (GPU likely already done by now)
    let h_coeffs = read_h_poly_result::<G>(gpu, h_pending).await?;

    #[cfg(feature = "timing")]
    eprintln!("[timing] h_poly: {:?}", t_h_start.elapsed());

    // H bucket data depends on h_coeffs
    let h_bd = compute_bucket_sorting::<G>(&h_coeffs[..pk.h.len()]);

    #[cfg(feature = "timing")]
    let t_msm_start = std::time::Instant::now();

    let (a_msm, b_g1_msm, l_msm, h_msm, b_g2_msm) = gpu_msm_batch_bytes::<G>(
        gpu,
        &ppk.a_bytes,
        a_bd,
        &ppk.b_g1_bytes,
        b1_bd,
        &ppk.l_bytes,
        l_bd,
        &ppk.h_bytes,
        h_bd,
        &ppk.b_g2_bytes,
        b2_bd,
    )
    .await?;

    #[cfg(feature = "timing")]
    eprintln!("[timing] msm_batch: {:?}", t_msm_start.elapsed());

    let mut proof_a = G::add_g1_proj(&G::affine_to_proj_g1(&pk.vk.alpha_g1), &a_msm);
    proof_a = G::add_g1_proj(&proof_a, &G::mul_g1_scalar(&pk.vk.delta_g1, &r));

    let mut proof_b = G::add_g2_proj(&G::affine_to_proj_g2(&pk.vk.beta_g2), &b_g2_msm);
    proof_b = G::add_g2_proj(&proof_b, &G::mul_g2_scalar(&pk.vk.delta_g2, &s));

    let mut proof_c = G::add_g1_proj(&l_msm, &h_msm);
    let mut b_g1 = G::add_g1_proj(&G::affine_to_proj_g1(&pk.vk.beta_g1), &b_g1_msm);
    b_g1 = G::add_g1_proj(&b_g1, &G::mul_g1_scalar(&pk.vk.delta_g1, &s));

    let c_shift_a = G::mul_g1_proj_scalar(&proof_a, &s);
    proof_c = G::add_g1_proj(&proof_c, &c_shift_a);

    let c_shift_b = G::mul_g1_proj_scalar(&b_g1, &r);
    proof_c = G::add_g1_proj(&proof_c, &c_shift_b);

    let mut rs = r;
    rs *= s;
    let rs_delta = G::mul_g1_scalar(&pk.vk.delta_g1, &rs);
    proof_c = G::sub_g1_proj(&proof_c, &rs_delta);

    Ok(bellman::groth16::Proof {
        a: G::proj_to_affine_g1(&proof_a),
        b: G::proj_to_affine_g2(&proof_b),
        c: G::proj_to_affine_g1(&proof_c),
    })
}

pub async fn create_proof<E, G, C, R>(
    circuit: C,
    pk: &bellman::groth16::Parameters<E>,
    ppk: &PreparedProvingKey<G>,
    gpu: &GpuContext<G>,
    rng: &mut R,
) -> Result<bellman::groth16::Proof<E>>
where
    E: pairing::MultiMillerLoop,
    C: bellman::Circuit<G::Scalar>,
    G: GpuCurve<
            Engine = E,
            Scalar = E::Fr,
            G1 = E::G1,
            G2 = E::G2,
            G1Affine = E::G1Affine,
            G2Affine = E::G2Affine,
        > + Send,
    R: RngCore,
{
    let r = G::Scalar::random(&mut *rng);
    let s = G::Scalar::random(&mut *rng);
    create_proof_with_fixed_randomness::<E, G, C>(circuit, pk, ppk, gpu, r, s).await
}

#[cfg(test)]
mod tests {
    use crate::bellman::Circuit;
    use blstrs::{Bls12, Scalar};
    use ff::Field;
    use group::Group;
    use masp_primitives::asset_type::AssetType;
    use masp_primitives::jubjub;
    use masp_primitives::sapling::{Diversifier, ProofGenerationKey};
    use masp_proofs::circuit::sapling::Output as SaplingOutputCircuit;
    use rand_core::OsRng;
    use std::time::Instant;

    use super::*;

    /// A simple dummy circuit that proves knowledge of a secret `x`
    /// such that `x^3 = y` (where `y` is public).
    struct DummyCircuit<Scalar: PrimeField> {
        pub x: Option<Scalar>,
        pub y: Option<Scalar>,
    }

    impl<Scalar: PrimeField> bellman::Circuit<Scalar> for DummyCircuit<Scalar> {
        fn synthesize<CS: bellman::ConstraintSystem<Scalar>>(
            self,
            cs: &mut CS,
        ) -> Result<(), bellman::SynthesisError> {
            // Allocate public input `y`
            let y_val = self.y;
            let y = cs.alloc_input(
                || "y",
                || y_val.ok_or(bellman::SynthesisError::AssignmentMissing),
            )?;

            // Allocate private input (witness) `x`
            let x_val = self.x;
            let x = cs.alloc(
                || "x",
                || x_val.ok_or(bellman::SynthesisError::AssignmentMissing),
            )?;

            // Intermediate constraint 1: x_sq = x * x
            let x_sq_val = x_val.map(|v| v.square());
            let x_sq = cs.alloc(
                || "x_sq",
                || x_sq_val.ok_or(bellman::SynthesisError::AssignmentMissing),
            )?;
            cs.enforce(|| "x * x = x_sq", |lc| lc + x, |lc| lc + x, |lc| lc + x_sq);

            // Intermediate constraint 2: y = x_sq * x  (which means y = x^3)
            cs.enforce(|| "x_sq * x = y", |lc| lc + x_sq, |lc| lc + x, |lc| lc + y);

            Ok(())
        }
    }

    #[tokio::test]
    async fn test_gpu_groth16_prover() {
        let mut rng = OsRng;

        // --------------------------------------------------------------------
        // 1. Trusted Setup
        // --------------------------------------------------------------------
        let setup_circuit = DummyCircuit::<Scalar> { x: None, y: None };

        // Use Bls12 as the Engine (E)
        let params =
            bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
                .expect("Failed to generate trusted setup parameters");

        // --------------------------------------------------------------------
        // 2. Initialize GPU Context
        // --------------------------------------------------------------------
        // Use Bls12 as the GpuCurve (G)
        let gpu_ctx = GpuContext::<Bls12>::new()
            .await
            .expect("Failed to initialize WebGPU context");

        // --------------------------------------------------------------------
        // 3. Generate GPU-Accelerated Proof
        // --------------------------------------------------------------------
        // Let's prove we know `x = 3` such that `x^3 = 27`.
        let x_value = Scalar::from(3u64);
        let y_value = Scalar::from(27u64);

        let circuit = DummyCircuit {
            x: Some(x_value),
            y: Some(y_value),
        };

        // Pass Bls12 for both the E and G generic parameters
        let ppk = prepare_proving_key::<Bls12, Bls12>(&params);
        let proof = create_proof::<Bls12, Bls12, _, _>(circuit, &params, &ppk, &gpu_ctx, &mut rng)
            .await
            .expect("Failed to generate Groth16 proof on GPU");

        // --------------------------------------------------------------------
        // 4. Verify Proof (CPU)
        // --------------------------------------------------------------------
        let pvk = bellman::groth16::prepare_verifying_key(&params.vk);

        // Our only public input is `y = 27`
        let public_inputs = vec![y_value];

        let is_valid = bellman::groth16::verify_proof(&pvk, &proof, &public_inputs)
            .expect("Failed during proof verification step");

        assert!(is_valid, "The generated Groth16 proof is invalid!");

        // --------------------------------------------------------------------
        // 5. Sanity Check: Invalid Proof Rejection
        // --------------------------------------------------------------------
        // Ensure that passing the wrong public input (e.g., y = 28) causes a rejection.
        let wrong_public_inputs = vec![Scalar::from(28u64)];
        let is_valid_wrong = bellman::groth16::verify_proof(&pvk, &proof, &wrong_public_inputs)
            .expect("Failed during proof verification step");

        assert!(
            !is_valid_wrong,
            "The verifier should reject a proof with tampered public inputs"
        );
    }

    #[test]
    fn test_cpu_groth16_prover_baseline() {
        let mut rng = OsRng;

        let setup_circuit = DummyCircuit::<Scalar> { x: None, y: None };
        let params =
            bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
                .expect("failed to generate trusted setup parameters");

        let x_value = Scalar::from(3u64);
        let y_value = Scalar::from(27u64);
        let circuit = DummyCircuit {
            x: Some(x_value),
            y: Some(y_value),
        };

        let proof = bellman::groth16::create_random_proof(circuit, &params, &mut rng)
            .expect("failed to create cpu proof");

        let pvk = bellman::groth16::prepare_verifying_key(&params.vk);
        let public_inputs = vec![y_value];
        let is_valid = bellman::groth16::verify_proof(&pvk, &proof, &public_inputs)
            .expect("verification failed");
        assert!(is_valid, "cpu proof should verify");
    }

    #[test]
    #[ignore = "benchmark-style test"]
    fn bench_cpu_msm_100k_like() {
        let mut rng = OsRng;
        let setup_circuit = DummyCircuit::<Scalar> { x: None, y: None };
        let params =
            bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
                .expect("failed to generate parameters");

        let n = 100_000usize.min(params.a.len());
        let mut scalars = Vec::with_capacity(n);
        for i in 0..n {
            scalars.push(Scalar::from((i as u64) + 1));
        }

        let t0 = Instant::now();
        let mut acc = <Bls12 as pairing::Engine>::G1::identity();
        for (b, s) in params.a.iter().take(n).zip(scalars.iter()) {
            acc += *b * *s;
        }
        let dt = t0.elapsed();
        eprintln!("CPU MSM n={n} took {:?}", dt);
        let _ = acc;
    }

    fn sample_sapling_output_circuit() -> SaplingOutputCircuit {
        let asset_type = AssetType::new(b"benchmark-asset").expect("asset type creation failed");
        let value_commitment = asset_type.value_commitment(42, jubjub::Fr::from(7u64));

        let pgk = ProofGenerationKey {
            ak: jubjub::SubgroupPoint::generator(),
            nsk: jubjub::Fr::from(11u64),
        };
        let vk = pgk.to_viewing_key();
        let mut payment_address = None;
        for d0 in 0u8..=255 {
            if let Some(addr) =
                vk.to_payment_address(Diversifier([d0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            {
                payment_address = Some(addr);
                break;
            }
        }
        let payment_address = payment_address.expect("failed to find a valid diversifier");

        SaplingOutputCircuit {
            value_commitment: Some(value_commitment),
            asset_identifier: asset_type.identifier_bits(),
            payment_address: Some(payment_address),
            commitment_randomness: Some(jubjub::Fr::from(13u64)),
            esk: Some(jubjub::Fr::from(17u64)),
        }
    }

    #[test]
    #[ignore = "benchmark-style test"]
    fn bench_cpu_sapling_output() {
        let mut rng = OsRng;
        let setup = SaplingOutputCircuit {
            value_commitment: None,
            asset_identifier: vec![None; 256],
            payment_address: None,
            commitment_randomness: None,
            esk: None,
        };
        let params = bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup, &mut rng)
            .expect("failed to generate sapling output parameters");

        let circuit = sample_sapling_output_circuit();
        let t0 = Instant::now();
        let proof = bellman::groth16::create_random_proof(circuit, &params, &mut rng)
            .expect("cpu sapling output proof failed");
        let dt = t0.elapsed();
        let _ = proof;
        eprintln!("CPU Sapling Output proof took {:?}", dt);
    }

    #[tokio::test]
    #[ignore = "benchmark-style test"]
    async fn bench_gpu_sapling_output() {
        let mut rng = OsRng;
        let setup = SaplingOutputCircuit {
            value_commitment: None,
            asset_identifier: vec![None; 256],
            payment_address: None,
            commitment_randomness: None,
            esk: None,
        };
        let params = bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup, &mut rng)
            .expect("failed to generate sapling output parameters");
        let gpu_ctx = GpuContext::<Bls12>::new()
            .await
            .expect("failed to initialize gpu");

        let circuit = sample_sapling_output_circuit();
        let ppk = prepare_proving_key::<Bls12, Bls12>(&params);
        let t0 = Instant::now();
        let _proof = create_proof::<Bls12, Bls12, _, _>(circuit, &params, &ppk, &gpu_ctx, &mut rng)
            .await
            .expect("gpu sapling output proof failed");
        let dt = t0.elapsed();
        eprintln!("GPU Sapling Output proof took {:?}", dt);
    }

    #[tokio::test]
    async fn test_gpu_msm_single_point_matches_cpu_g1() {
        let mut rng = OsRng;
        let setup_circuit = DummyCircuit::<Scalar> { x: None, y: None };
        let params =
            bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
                .expect("failed to generate parameters");
        let gpu_ctx = GpuContext::<Bls12>::new()
            .await
            .expect("failed to initialize gpu");

        let base = params.a[0];
        let scalar = Scalar::from(1u64);

        let gpu = gpu_msm_g1::<Bls12>(&gpu_ctx, &[base], &[scalar])
            .await
            .expect("gpu msm failed");
        let gpu_affine = Bls12::proj_to_affine_g1(&gpu);
        assert_eq!(gpu_affine, base);
    }

    #[tokio::test]
    async fn test_gpu_msm_single_point_matches_cpu_g2() {
        let mut rng = OsRng;
        let setup_circuit = DummyCircuit::<Scalar> { x: None, y: None };
        let params =
            bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
                .expect("failed to generate parameters");
        let gpu_ctx = GpuContext::<Bls12>::new()
            .await
            .expect("failed to initialize gpu");

        let base = params.b_g2[0];
        let scalar = Scalar::from(1u64);

        let gpu = gpu_msm_g2::<Bls12>(&gpu_ctx, &[base], &[scalar])
            .await
            .expect("gpu msm failed");
        let gpu_affine = Bls12::proj_to_affine_g2(&gpu);
        assert_eq!(gpu_affine, base);
    }

    #[tokio::test]
    async fn test_proof_abc_match_cpu_reference() {
        let mut rng = OsRng;
        let setup_circuit = DummyCircuit::<Scalar> { x: None, y: None };
        let params =
            bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
                .expect("failed to generate parameters");
        let gpu_ctx = GpuContext::<Bls12>::new()
            .await
            .expect("failed to initialize gpu");

        let x_value = Scalar::from(3u64);
        let y_value = Scalar::from(27u64);
        let circuit = DummyCircuit {
            x: Some(x_value),
            y: Some(y_value),
        };

        let r = Scalar::ZERO;
        let s = Scalar::ZERO;

        let ppk = prepare_proving_key::<Bls12, Bls12>(&params);
        let gpu_proof = create_proof_with_fixed_randomness::<Bls12, Bls12, _>(
            DummyCircuit {
                x: Some(x_value),
                y: Some(y_value),
            },
            &params,
            &ppk,
            &gpu_ctx,
            r,
            s,
        )
        .await
        .expect("gpu proof creation failed");

        let cpu_proof = bellman::groth16::create_proof(circuit, &params, r, s)
            .expect("cpu proof creation failed");

        let gpu_a: blstrs::G1Projective = gpu_proof.a.into();
        let cpu_a: blstrs::G1Projective = cpu_proof.a.into();
        let gpu_b: blstrs::G2Projective = gpu_proof.b.into();
        let cpu_b: blstrs::G2Projective = cpu_proof.b.into();
        let gpu_c: blstrs::G1Projective = gpu_proof.c.into();
        let cpu_c: blstrs::G1Projective = cpu_proof.c.into();

        let a_ok = gpu_a == cpu_a;
        let b_ok = gpu_b == cpu_b;
        let c_ok = gpu_c == cpu_c;
        assert!(
            a_ok && b_ok && c_ok,
            "proof components mismatch: a={a_ok} b={b_ok} c={c_ok}"
        );
    }

    #[tokio::test]
    async fn test_h_msm_matches_cpu_for_dummy_circuit() {
        let mut rng = OsRng;
        let setup_circuit = DummyCircuit::<Scalar> { x: None, y: None };
        let params =
            bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
                .expect("failed to generate parameters");
        let gpu_ctx = GpuContext::<Bls12>::new()
            .await
            .expect("failed to initialize gpu");

        let x_value = Scalar::from(3u64);
        let y_value = Scalar::from(27u64);

        let mut cs = GpuConstraintSystem::<Bls12>::new();
        DummyCircuit {
            x: Some(x_value),
            y: Some(y_value),
        }
        .synthesize(&mut cs)
        .expect("synthesis failed");

        for i in 0..cs.inputs.len() {
            cs.a_lcs.push(vec![(
                bellman::Variable::new_unchecked(bellman::Index::Input(i)),
                Scalar::ONE,
            )]);
            cs.b_lcs.push(Vec::new());
            cs.c_lcs.push(Vec::new());
        }

        let num_constraints = cs.a_lcs.len();
        let n = num_constraints.next_power_of_two();
        let mut a_values = vec![Scalar::ZERO; n];
        let mut b_values = vec![Scalar::ZERO; n];
        let mut c_values = vec![Scalar::ZERO; n];
        for i in 0..num_constraints {
            a_values[i] = eval_lc(&cs.a_lcs[i], &cs.inputs, &cs.aux);
            b_values[i] = eval_lc(&cs.b_lcs[i], &cs.inputs, &cs.aux);
            c_values[i] = eval_lc(&cs.c_lcs[i], &cs.inputs, &cs.aux);
        }

        let h_coeffs = compute_h_poly(&gpu_ctx, &a_values, &b_values, &c_values)
            .await
            .expect("compute_h_poly failed");

        let gpu_h = gpu_msm_g1::<Bls12>(&gpu_ctx, &params.h, &h_coeffs[..params.h.len()])
            .await
            .expect("gpu h msm failed");

        let mut cpu_h = <Bls12 as pairing::Engine>::G1::identity();
        for (b, s) in params.h.iter().zip(h_coeffs.iter()) {
            cpu_h += b * *s;
        }

        assert_eq!(
            Bls12::proj_to_affine_g1(&gpu_h),
            Bls12::proj_to_affine_g1(&cpu_h)
        );
    }

    #[tokio::test]
    async fn test_ab_msm_match_cpu_for_dummy_circuit() {
        let mut rng = OsRng;
        let setup_circuit = DummyCircuit::<Scalar> { x: None, y: None };
        let params =
            bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
                .expect("failed to generate parameters");
        let gpu_ctx = GpuContext::<Bls12>::new()
            .await
            .expect("failed to initialize gpu");

        let x_value = Scalar::from(3u64);
        let y_value = Scalar::from(27u64);

        let mut cs = GpuConstraintSystem::<Bls12>::new();
        DummyCircuit {
            x: Some(x_value),
            y: Some(y_value),
        }
        .synthesize(&mut cs)
        .expect("synthesis failed");

        let mut a_assignment = cs.inputs.clone();
        for (i, s) in cs.aux.iter().enumerate() {
            if cs.a_aux_density[i] {
                a_assignment.push(*s);
            }
        }
        let b_assignment = dense_assignment_from_masks(
            &cs.inputs,
            &cs.aux,
            &cs.b_input_density,
            &cs.b_aux_density,
        );

        let gpu_a = gpu_msm_g1::<Bls12>(&gpu_ctx, &params.a, &a_assignment)
            .await
            .expect("gpu a msm failed");
        let gpu_b_g1 = gpu_msm_g1::<Bls12>(&gpu_ctx, &params.b_g1, &b_assignment)
            .await
            .expect("gpu b_g1 msm failed");
        let gpu_b_g2 = gpu_msm_g2::<Bls12>(&gpu_ctx, &params.b_g2, &b_assignment)
            .await
            .expect("gpu b_g2 msm failed");

        let mut cpu_a = <Bls12 as pairing::Engine>::G1::identity();
        for (b, s) in params.a.iter().zip(a_assignment.iter()) {
            cpu_a += b * *s;
        }
        let mut cpu_b_g1 = <Bls12 as pairing::Engine>::G1::identity();
        for (b, s) in params.b_g1.iter().zip(b_assignment.iter()) {
            cpu_b_g1 += b * *s;
        }
        let mut cpu_b_g2 = <Bls12 as pairing::Engine>::G2::identity();
        for (b, s) in params.b_g2.iter().zip(b_assignment.iter()) {
            cpu_b_g2 += b * *s;
        }

        let a_ok = Bls12::proj_to_affine_g1(&gpu_a) == Bls12::proj_to_affine_g1(&cpu_a);
        let b1_ok = Bls12::proj_to_affine_g1(&gpu_b_g1) == Bls12::proj_to_affine_g1(&cpu_b_g1);
        let b2_ok = Bls12::proj_to_affine_g2(&gpu_b_g2) == Bls12::proj_to_affine_g2(&cpu_b_g2);
        assert!(
            a_ok && b1_ok && b2_ok,
            "ab msm mismatch: a={a_ok} b_g1={b1_ok} b_g2={b2_ok}"
        );
    }

    #[tokio::test]
    async fn test_h_component_matches_cpu_when_r_s_zero() {
        let mut rng = OsRng;
        let setup_circuit = DummyCircuit::<Scalar> { x: None, y: None };
        let params =
            bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
                .expect("failed to generate parameters");
        let gpu_ctx = GpuContext::<Bls12>::new()
            .await
            .expect("failed to initialize gpu");

        let x_value = Scalar::from(3u64);
        let y_value = Scalar::from(27u64);

        let mut cs = GpuConstraintSystem::<Bls12>::new();
        DummyCircuit {
            x: Some(x_value),
            y: Some(y_value),
        }
        .synthesize(&mut cs)
        .expect("synthesis failed");

        for i in 0..cs.inputs.len() {
            cs.a_lcs.push(vec![(
                bellman::Variable::new_unchecked(bellman::Index::Input(i)),
                Scalar::ONE,
            )]);
            cs.b_lcs.push(Vec::new());
            cs.c_lcs.push(Vec::new());
        }

        let num_constraints = cs.a_lcs.len();
        let n = num_constraints.next_power_of_two();
        let mut a_values = vec![Scalar::ZERO; n];
        let mut b_values = vec![Scalar::ZERO; n];
        let mut c_values = vec![Scalar::ZERO; n];
        for i in 0..num_constraints {
            a_values[i] = eval_lc(&cs.a_lcs[i], &cs.inputs, &cs.aux);
            b_values[i] = eval_lc(&cs.b_lcs[i], &cs.inputs, &cs.aux);
            c_values[i] = eval_lc(&cs.c_lcs[i], &cs.inputs, &cs.aux);
        }

        let h_coeffs = compute_h_poly(&gpu_ctx, &a_values, &b_values, &c_values)
            .await
            .expect("compute_h_poly failed");
        let gpu_h = gpu_msm_g1::<Bls12>(&gpu_ctx, &params.h, &h_coeffs[..params.h.len()])
            .await
            .expect("gpu h msm failed");
        let gpu_l = gpu_msm_g1::<Bls12>(&gpu_ctx, &params.l, &cs.aux)
            .await
            .expect("gpu l msm failed");

        let cpu_proof = bellman::groth16::create_proof(
            DummyCircuit {
                x: Some(x_value),
                y: Some(y_value),
            },
            &params,
            Scalar::ZERO,
            Scalar::ZERO,
        )
        .expect("cpu proof failed");

        let cpu_c: blstrs::G1Projective = cpu_proof.c.into();
        let mut cpu_l = <Bls12 as pairing::Engine>::G1::identity();
        for (b, s) in params.l.iter().zip(cs.aux.iter()) {
            cpu_l += b * *s;
        }

        let mut cpu_h = cpu_c;
        cpu_h -= cpu_l;

        assert_eq!(
            Bls12::proj_to_affine_g1(&gpu_l),
            Bls12::proj_to_affine_g1(&cpu_l),
            "l component mismatch"
        );
        assert_eq!(
            Bls12::proj_to_affine_g1(&gpu_h),
            Bls12::proj_to_affine_g1(&cpu_h),
            "h component mismatch"
        );
    }
}
