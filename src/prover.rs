// TODO: implement a form of efficient batching of GPU operations.
// loading data in and out of the GPU is slow. we want to batch as many ops as possible,
// without using a stupid amount of VRAM (perhaps we need to check
// how much VRAM the host has).

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

    // CPU Combination logic (standard Pippenger loop over window returns)
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

    // Padding domain size to 2N avoids cyclic wrap-around
    // when multiplying two polynomials of degree (N-1) -> resulting degree 2N - 2
    let domain_size = n * 2;

    // 1. CPU PRE-COMPUTES CONSTANT FACTORS (Takes < 1 ms)
    let omega_n = G::root_of_unity(n);
    let omega_n_inv = omega_n.invert().unwrap();
    let omega_2n = G::root_of_unity(domain_size);
    let omega_2n_inv = omega_2n.invert().unwrap();

    let n_inv = G::Scalar::from(n as u64).invert().unwrap();
    let coset_generator = G::Scalar::MULTIPLICATIVE_GENERATOR;
    let coset_inv = coset_generator.invert().unwrap();
    let domain_inv = G::Scalar::from(domain_size as u64).invert().unwrap();

    let mut inv_twiddles_n = vec![G::Scalar::ONE; n];
    let mut fwd_twiddles_2n = vec![G::Scalar::ONE; domain_size];
    let mut inv_twiddles_2n = vec![G::Scalar::ONE; domain_size];
    let mut shifts = vec![G::Scalar::ONE; n];
    let mut inv_shifts = vec![G::Scalar::ONE; n];

    for i in 1..domain_size {
        if i < n {
            inv_twiddles_n[i] = inv_twiddles_n[i - 1] * omega_n_inv;
            shifts[i] = shifts[i - 1] * coset_generator;
            inv_shifts[i] = inv_shifts[i - 1] * coset_inv;
        }
        fwd_twiddles_2n[i] = fwd_twiddles_2n[i - 1] * omega_2n;
        inv_twiddles_2n[i] = inv_twiddles_2n[i - 1] * omega_2n_inv;
    }

    // Embed `n_inv` and `domain_inv` directly into the shifts so the GPU avoids
    // doing a separate scalar multiplication pass.
    for i in 0..n {
        shifts[i] *= n_inv;
        inv_shifts[i] *= domain_inv;
    }

    let g_to_n = coset_generator.pow([n as u64]);
    let z_even_inv = (g_to_n - G::Scalar::ONE).invert().unwrap();
    let z_odd_inv = (-g_to_n - G::Scalar::ONE).invert().unwrap();
    let z_invs = vec![z_even_inv, z_odd_inv];

    // 2. CPU ZERO-PADDING (Sizes up to 2N before sending)
    let mut a_coeffs = a_values.to_vec();
    a_coeffs.resize(domain_size, G::Scalar::ZERO);
    let mut b_coeffs = b_values.to_vec();
    b_coeffs.resize(domain_size, G::Scalar::ZERO);
    let mut c_coeffs = c_values.to_vec();
    c_coeffs.resize(domain_size, G::Scalar::ZERO);

    // 3. UPLOAD ALL BUFFERS TO VRAM (One-way trip)
    let a_buf = gpu.create_storage_buffer("A", &marshal_scalars::<G>(&a_coeffs));
    let b_buf = gpu.create_storage_buffer("B", &marshal_scalars::<G>(&b_coeffs));
    let c_buf = gpu.create_storage_buffer("C", &marshal_scalars::<G>(&c_coeffs));
    let h_buf = gpu.create_empty_buffer("H", (domain_size * 32) as u64);

    let tw_inv_n_buf = gpu.create_storage_buffer("TwInvN", &marshal_scalars::<G>(&inv_twiddles_n));
    let tw_fwd_2n_buf =
        gpu.create_storage_buffer("TwFwd2N", &marshal_scalars::<G>(&fwd_twiddles_2n));
    let tw_inv_2n_buf =
        gpu.create_storage_buffer("TwInv2N", &marshal_scalars::<G>(&inv_twiddles_2n));
    let shifts_buf = gpu.create_storage_buffer("Shifts", &marshal_scalars::<G>(&shifts));
    let inv_shifts_buf = gpu.create_storage_buffer("InvShifts", &marshal_scalars::<G>(&inv_shifts));
    let z_invs_buf = gpu.create_storage_buffer("ZInvs", &marshal_scalars::<G>(&z_invs));

    // 4. DISPATCH 100% VRAM-BOUND GPU COMPUTE CHAIN
    // Step A: iNTT(N) to get coefficients
    gpu.execute_ntt(&a_buf, &tw_inv_n_buf, n as u32);
    gpu.execute_ntt(&b_buf, &tw_inv_n_buf, n as u32);
    gpu.execute_ntt(&c_buf, &tw_inv_n_buf, n as u32);

    // Step B: Shift evaluations by Coset Generator
    gpu.execute_coset_shift(&a_buf, &shifts_buf, n as u32);
    gpu.execute_coset_shift(&b_buf, &shifts_buf, n as u32);
    gpu.execute_coset_shift(&c_buf, &shifts_buf, n as u32);

    // Step C: NTT(2N) to get shifted evaluations over padded domain
    gpu.execute_ntt(&a_buf, &tw_fwd_2n_buf, domain_size as u32);
    gpu.execute_ntt(&b_buf, &tw_fwd_2n_buf, domain_size as u32);
    gpu.execute_ntt(&c_buf, &tw_fwd_2n_buf, domain_size as u32);

    // Step D: Evaluate H(X) pointwise
    gpu.execute_pointwise_poly(
        &a_buf,
        &b_buf,
        &c_buf,
        &h_buf,
        &z_invs_buf,
        domain_size as u32,
    );

    // Step E: iNTT(2N) to get back H(X) coefficients
    gpu.execute_ntt(&h_buf, &tw_inv_2n_buf, domain_size as u32);

    // Step F: Reverse the Coset Shift on the resulting N coefficients
    gpu.execute_coset_shift(&h_buf, &inv_shifts_buf, n as u32);

    // 5. FINALLY PULL THE RESULT OFF THE GPU
    // We only read back the first N elements (H drops the top half mathematically)
    let h_bytes = gpu
        .read_buffer(&h_buf, (n * 32) as wgpu::BufferAddress)
        .await?;

    let mut h_poly = vec![G::Scalar::ZERO; n];
    for (i, chunk) in h_bytes.chunks_exact(32).enumerate() {
        h_poly[i] = G::deserialize_scalar(chunk)?;
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

    // Groth16 naturally drops the highest degree component, meaning pk.h_query.len() = n - 1.
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
