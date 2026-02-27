// TODO: implement a form of efficient batching of GPU operations.
// loading data in and out of the GPU is slow. we want to batch as many ops as possible,
// without using a stupid amount of VRAM (perhaps we need to check
// how much VRAM the host has). if we're constantly calling `gpu_msm_g1`,
// `gpu_msm_g2`, and `gpu_fft` with small batches, we pay the cost
// of GPU data transfer, and won't gain much in terms of proving speed.

use anyhow::Result;
use ff::{Field, PrimeField};
use rand_core::RngCore;

use crate::bucket::{BucketData, compute_bucket_sorting};
use crate::gpu::{GpuContext, curve::GpuCurve};

// TODO: replace with [`bellman::groth16::Proof`]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof<G: GpuCurve> {
    pub a: G::G1Affine,
    pub b: G::G2Affine,
    pub c: G::G1Affine,
}

// TODO: in `create_proof`, replace this type with a generic
// type that implements the trait [`bellman::groth16::ParameterSource`]
pub struct ProvingKey<G: GpuCurve> {
    pub alpha_g1: G::G1Affine,
    pub beta_g1: G::G1Affine,
    pub beta_g2: G::G2Affine,
    pub delta_g1: G::G1Affine,
    pub delta_g2: G::G2Affine,

    pub a_query: Vec<G::G1Affine>,
    pub b_query_g1: Vec<G::G1Affine>,
    pub b_query_g2: Vec<G::G2Affine>,
    pub h_query: Vec<G::G1Affine>,
    pub l_query: Vec<G::G1Affine>,
}

struct GpuConstraintSystem<G: GpuCurve> {
    inputs: Vec<G::Scalar>,
    aux: Vec<G::Scalar>,
    a_lcs: Vec<Vec<(usize, G::Scalar)>>,
    b_lcs: Vec<Vec<(usize, G::Scalar)>>,
    c_lcs: Vec<Vec<(usize, G::Scalar)>>,
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
            a_lcs: Vec::new(),
            b_lcs: Vec::new(),
            c_lcs: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<G: GpuCurve> bellman::ConstraintSystem<G::Scalar> for GpuConstraintSystem<G> {
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
        self.a_lcs.push(lc_to_vec(a_lc));
        self.b_lcs.push(lc_to_vec(b_lc));
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

fn lc_to_vec<S: PrimeField>(lc: bellman::LinearCombination<S>) -> Vec<(usize, S)> {
    lc.as_ref()
        .iter()
        .map(|(var, coeff)| {
            let idx = match var.get_unchecked() {
                bellman::Index::Input(i) => i,
                bellman::Index::Aux(i) => i,
            };
            (idx, *coeff)
        })
        .collect()
}

fn marshal_scalars<G: GpuCurve>(scalars: &[G::Scalar]) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(scalars.len() * 32);
    for s in scalars {
        buffer.extend_from_slice(&G::serialize_scalar(s));
    }
    buffer
}

async fn gpu_fft<G: GpuCurve>(
    gpu: &GpuContext<G>,
    data: &mut [G::Scalar],
    twiddles: &[G::Scalar],
) -> Result<()> {
    let data_bytes = marshal_scalars::<G>(data);
    let twiddles_bytes = marshal_scalars::<G>(twiddles);
    let data_buf = gpu.create_storage_buffer("NTT Data", &data_bytes);
    let twiddles_buf = gpu.create_storage_buffer("NTT Twiddles", &twiddles_bytes);
    gpu.execute_ntt(&data_buf, &twiddles_buf, data.len() as u32);
    let result_bytes = gpu
        .read_buffer(&data_buf, (data.len() * 32) as wgpu::BufferAddress)
        .await?;
    for (i, chunk) in result_bytes.chunks_exact(32).enumerate() {
        data[i] = G::deserialize_scalar(chunk)?;
    }
    Ok(())
}

async fn gpu_msm_g1<G: GpuCurve>(
    gpu: &GpuContext<G>,
    bases: &[G::G1Affine],
    scalars: &[G::Scalar],
) -> Result<G::G1Projective> {
    let bd: BucketData = compute_bucket_sorting::<G>(scalars);
    if bd.num_active_buckets == 0 {
        return Ok(G::g1_identity());
    }

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

    let result_bytes = gpu
        .read_buffer(&sums_buf, (bd.num_windows * 144) as u64)
        .await?;
    let mut result = G::g1_identity();

    for (i, chunk) in result_bytes.chunks_exact(144).enumerate().rev() {
        if i != (bd.num_windows - 1) as usize {
            for _ in 0..G::bucket_width() {
                result = G::add_g1_proj(&result, &result);
            }
        }
        let w_sum = G::deserialize_g1(chunk)?;
        result = G::add_g1_proj(&result, &w_sum);
    }
    Ok(result)
}

async fn gpu_msm_g2<G: GpuCurve>(
    gpu: &GpuContext<G>,
    bases: &[G::G2Affine],
    scalars: &[G::Scalar],
) -> Result<G::G2Projective> {
    let bd: BucketData = compute_bucket_sorting::<G>(scalars);
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
    let mut result = G::g2_identity();

    for (i, chunk) in result_bytes.chunks_exact(288).enumerate().rev() {
        if i != (bd.num_windows - 1) as usize {
            for _ in 0..G::bucket_width() {
                result = G::add_g2_proj(&result, &result);
            }
        }
        let w_sum = G::deserialize_g2(chunk)?;
        result = G::add_g2_proj(&result, &w_sum);
    }
    Ok(result)
}

fn eval_lc<S: PrimeField>(lc: &[(usize, S)], inputs: &[S], aux: &[S]) -> S {
    let mut res = S::ZERO;
    for &(idx, coeff) in lc {
        let val = if idx < inputs.len() {
            inputs[idx]
        } else {
            aux[idx - inputs.len()]
        };
        let mut term = val;
        term.mul_assign(&coeff);
        res.add_assign(&term);
    }
    res
}

async fn compute_h_poly<G: GpuCurve>(
    gpu: &GpuContext<G>,
    a_values: &[G::Scalar],
    b_values: &[G::Scalar],
    c_values: &[G::Scalar],
) -> Result<Vec<G::Scalar>> {
    let n = a_values.len().next_power_of_two();

    let domain_size = n * 2;

    // Step 1: Interpolate A, B, C back to standard coefficients using iNTT(N)
    let omega_n = G::root_of_unity(n);
    let omega_n_inv = omega_n.invert().unwrap();

    let mut inv_twiddles_n = vec![G::Scalar::ONE; n];
    for i in 1..n {
        inv_twiddles_n[i] = inv_twiddles_n[i - 1] * omega_n_inv;
    }

    let mut a_coeffs = a_values.to_vec();
    a_coeffs.resize(n, G::Scalar::ZERO);
    let mut b_coeffs = b_values.to_vec();
    b_coeffs.resize(n, G::Scalar::ZERO);
    let mut c_coeffs = c_values.to_vec();
    c_coeffs.resize(n, G::Scalar::ZERO);

    gpu_fft(gpu, &mut a_coeffs, &inv_twiddles_n).await?;
    gpu_fft(gpu, &mut b_coeffs, &inv_twiddles_n).await?;
    gpu_fft(gpu, &mut c_coeffs, &inv_twiddles_n).await?;

    // Step 2: Apply coset shift to evaluate Z_H safely without Division-by-Zero panic
    let n_inv = G::Scalar::from(n as u64).invert().unwrap();

    let coset_generator = G::Scalar::MULTIPLICATIVE_GENERATOR;
    let mut shift = G::Scalar::ONE;

    for i in 0..n {
        let mult = n_inv * shift;
        a_coeffs[i] *= mult;
        b_coeffs[i] *= mult;
        c_coeffs[i] *= mult;
        shift *= coset_generator;
    }

    // Step 3: Pad the shifted coefficients to evaluate on the 2N domain
    a_coeffs.resize(domain_size, G::Scalar::ZERO);
    b_coeffs.resize(domain_size, G::Scalar::ZERO);
    c_coeffs.resize(domain_size, G::Scalar::ZERO);

    let omega_2n = G::root_of_unity(domain_size);
    let mut fwd_twiddles_2n = vec![G::Scalar::ONE; domain_size];
    for i in 1..domain_size {
        fwd_twiddles_2n[i] = fwd_twiddles_2n[i - 1] * omega_2n;
    }

    gpu_fft(gpu, &mut a_coeffs, &fwd_twiddles_2n).await?;
    gpu_fft(gpu, &mut b_coeffs, &fwd_twiddles_2n).await?;
    gpu_fft(gpu, &mut c_coeffs, &fwd_twiddles_2n).await?;

    // Step 4: Pointwise Polynomial Equation H(X) = (A(X) * B(X) - C(X)) / Z_H(X)
    let mut h_evals = vec![G::Scalar::ZERO; domain_size];
    let g_to_n = coset_generator.pow([n as u64]);

    // Mathematical Optimization: Precompute the vanishing evaluations
    // Z_H(X) = X^N - 1, substituting shifted roots -> Z_H = g^N * (-1)^i - 1
    let z_even = g_to_n - G::Scalar::ONE;
    let z_odd = -g_to_n - G::Scalar::ONE;

    let z_even_inv = z_even.invert().unwrap();
    let z_odd_inv = z_odd.invert().unwrap();

    for i in 0..domain_size {
        let z_inv = if i % 2 == 0 { z_even_inv } else { z_odd_inv };
        h_evals[i] = ((a_coeffs[i] * b_coeffs[i]) - c_coeffs[i]) * z_inv;
    }

    // Step 5: Convert the H(X) evaluations back to standard coefficients using iNTT(2N)
    let omega_2n_inv = omega_2n.invert().unwrap();
    let mut inv_twiddles_2n = vec![G::Scalar::ONE; domain_size];
    for i in 1..domain_size {
        inv_twiddles_2n[i] = inv_twiddles_2n[i - 1] * omega_2n_inv;
    }

    gpu_fft(gpu, &mut h_evals, &inv_twiddles_2n).await?;

    // Step 6: Remove the coset shift to get standard H coefficients (We only need the first N)
    let domain_size_inv = G::Scalar::from(domain_size as u64).invert().unwrap();
    let shift_inv = coset_generator.invert().unwrap();
    let mut current_shift = G::Scalar::ONE;

    let mut h_poly = vec![G::Scalar::ZERO; n];
    for i in 0..n {
        h_poly[i] = h_evals[i] * domain_size_inv * current_shift;
        current_shift *= shift_inv;
    }

    Ok(h_poly)
}

pub async fn create_proof<C, G, R>(
    circuit: C,
    pk: &ProvingKey<G>,
    gpu: &GpuContext<G>,
    rng: &mut R,
) -> Result<Proof<G>>
where
    C: bellman::Circuit<G::Scalar>,
    G: GpuCurve,
    R: RngCore,
{
    let mut cs = GpuConstraintSystem::<G>::new();
    circuit
        .synthesize(&mut cs)
        .map_err(|e| anyhow::anyhow!("circuit synthesis failed: {:?}", e))?;

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

    let h_coeffs = compute_h_poly(gpu, &a_values, &b_values, &c_values).await?;

    let mut full_assignment = cs.inputs.clone();
    full_assignment.extend(cs.aux.clone());

    let a_msm = gpu_msm_g1(gpu, &pk.a_query, &full_assignment).await?;
    let b_g1_msm = gpu_msm_g1(gpu, &pk.b_query_g1, &full_assignment).await?;
    let l_msm = gpu_msm_g1(gpu, &pk.l_query, &cs.aux).await?;

    let h_msm = gpu_msm_g1(gpu, &pk.h_query, &h_coeffs[..pk.h_query.len()]).await?;

    let b_g2_msm = gpu_msm_g2(gpu, &pk.b_query_g2, &full_assignment).await?;

    let r = G::Scalar::random(&mut *rng);
    let s = G::Scalar::random(&mut *rng);

    let mut proof_a = G::add_g1_proj(&G::affine_to_proj_g1(&pk.alpha_g1), &a_msm);
    proof_a = G::add_g1_proj(&proof_a, &G::mul_g1_scalar(&pk.delta_g1, &r));

    let mut proof_b = G::add_g2_proj(&G::affine_to_proj_g2(&pk.beta_g2), &b_g2_msm);
    proof_b = G::add_g2_proj(&proof_b, &G::mul_g2_scalar(&pk.delta_g2, &s));

    let mut proof_c = G::add_g1_proj(&l_msm, &h_msm);
    let mut b_g1 = G::add_g1_proj(&G::affine_to_proj_g1(&pk.beta_g1), &b_g1_msm);
    b_g1 = G::add_g1_proj(&b_g1, &G::mul_g1_scalar(&pk.delta_g1, &s));

    let c_shift_a = G::mul_g1_proj_scalar(&proof_a, &s);
    proof_c = G::add_g1_proj(&proof_c, &c_shift_a);

    let c_shift_b = G::mul_g1_proj_scalar(&b_g1, &r);
    proof_c = G::add_g1_proj(&proof_c, &c_shift_b);

    let mut rs = r;
    rs *= s;
    let rs_delta = G::mul_g1_scalar(&pk.delta_g1, &rs);
    proof_c = G::sub_g1_proj(&proof_c, &rs_delta);

    Ok(Proof {
        a: G::proj_to_affine_g1(&proof_a),
        b: G::proj_to_affine_g2(&proof_b),
        c: G::proj_to_affine_g1(&proof_c),
    })
}
