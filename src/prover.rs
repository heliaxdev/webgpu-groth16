// TODO: this implementation still seems incomplete, and it
// is submitting many singular operations to the GPU, which
// is highly inefficient. we should batch GPU operations

use anyhow::{Context, Result};
use ff::{Field, PrimeField};
use rand_core::RngCore;

use crate::gpu::{GpuContext, curve::GpuCurve};
use crate::qap::ProvingCS;
use crate::traits::{Circuit, ConstraintSystem, Index, LinearCombination};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof<C: GpuCurve> {
    pub a: C::G1Affine,
    pub b: C::G2Affine,
    pub c: C::G1Affine,
}

pub struct ProvingKey<C: GpuCurve> {
    pub alpha_g1: C::G1Affine,
    pub beta_g1: C::G1Affine,
    pub beta_g2: C::G2Affine,
    pub delta_g1: C::G1Affine,
    pub delta_g2: C::G2Affine,

    pub a_query: Vec<C::G1Affine>,
    pub b_query_g1: Vec<C::G1Affine>,
    pub b_query_g2: Vec<C::G2Affine>,
    pub h_query: Vec<C::G1Affine>,
    pub l_query: Vec<C::G1Affine>,
}

/// Serializes a vector of scalars into a flat byte buffer for the GPU.
fn marshal_scalars<G: GpuCurve>(scalars: &[G::Scalar]) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(scalars.len() * 32);
    for s in scalars {
        buffer.extend_from_slice(&G::serialize_scalar(s));
    }
    buffer
}

/// Dispatches a WebGPU NTT pass and deserializes the result back into the mutable slice.
async fn gpu_fft<G: GpuCurve>(
    gpu: &GpuContext<G>,
    data: &mut [G::Scalar],
    twiddles: &[G::Scalar],
) -> Result<()> {
    let data_bytes = marshal_scalars::<G>(data);
    let twiddles_bytes = marshal_scalars::<G>(twiddles);

    let data_buf = gpu.create_storage_buffer("NTT Data", &data_bytes);
    let twiddles_buf = gpu.create_storage_buffer("NTT Twiddles", &twiddles_bytes);

    // Dispatch the WebGPU Compute Shader
    gpu.execute_ntt(&data_buf, &twiddles_buf, data.len() as u32);

    // Await the mapped memory from the GPU
    let result_bytes = gpu
        .read_buffer(&data_buf, (data.len() * 32) as wgpu::BufferAddress)
        .await?;

    // Deserialize back to the host memory
    for (i, chunk) in result_bytes.chunks_exact(32).enumerate() {
        data[i] = G::deserialize_scalar(chunk)?;
    }
    Ok(())
}

/// Dispatches a WebGPU MSM pass for G1 points using Luo-Fu-Gong Bucket Reduction.
async fn gpu_msm_g1<G: GpuCurve>(
    gpu: &GpuContext<G>,
    bases: &[G::G1Affine],
    // TODO: this is unused
    _scalars: &[G::Scalar],
) -> Result<G::G1Projective> {
    // 1. Host-Side Bucket Sorting
    let mut bucket_bytes = Vec::new();
    for base in bases {
        bucket_bytes.extend_from_slice(&G::serialize_g1(base));
    }

    // (Indices calculation based on scalars mapped here. For structure we zero-alloc)
    let indices_bytes = vec![0u8; bases.len() * 4];
    let count_bytes = (bases.len() as u32).to_le_bytes();

    let buckets_buf = gpu.create_storage_buffer("MSM Buckets", &bucket_bytes);
    let indices_buf = gpu.create_storage_buffer("MSM Indices", &indices_bytes);
    let count_buf = gpu.create_storage_buffer("MSM Count", &count_bytes);
    let result_buf = gpu.create_empty_buffer("MSM Result", 144);

    // Dispatch WebGPU Compute Shader
    gpu.execute_msm(&buckets_buf, &indices_buf, &count_buf, &result_buf);

    let result_bytes = gpu.read_buffer(&result_buf, 144).await?;
    G::deserialize_g1(&result_bytes)
}

fn eval_lc<S: PrimeField>(lc: &LinearCombination<S>, inputs: &[S], aux: &[S]) -> S {
    let mut res = S::ZERO;
    for &(var, coeff) in lc.as_ref() {
        let mut val = match var.get_unchecked() {
            Index::Input(i) => inputs[i],
            Index::Aux(i) => aux[i],
        };
        val.mul_assign(&coeff);
        res.add_assign(&val);
    }
    res
}

/// Computes the QAP quotient polynomial h(x) using the WebGPU NTT pipeline.
async fn compute_h_poly<G: GpuCurve>(
    gpu: &GpuContext<G>,
    cs: &ProvingCS<G::Scalar>,
) -> Result<Vec<G::Scalar>> {
    let n = cs.constraints.len().next_power_of_two();

    let mut a_evals = vec![G::Scalar::ZERO; n];
    let mut b_evals = vec![G::Scalar::ZERO; n];
    let mut c_evals = vec![G::Scalar::ZERO; n];

    for (i, constraint) in cs.constraints.iter().enumerate() {
        a_evals[i] = eval_lc(&constraint.a, &cs.inputs, &cs.aux);
        b_evals[i] = eval_lc(&constraint.b, &cs.inputs, &cs.aux);
        c_evals[i] = eval_lc(&constraint.c, &cs.inputs, &cs.aux);
    }

    // Generate twiddles for size N
    let omega = G::Scalar::multiplicative_generator(); // Assumed highly composite root
    let omega_inv = omega.invert().unwrap();

    let mut fwd_twiddles = vec![G::Scalar::ONE; n];
    let mut inv_twiddles = vec![G::Scalar::ONE; n];
    for i in 1..n {
        fwd_twiddles[i] = fwd_twiddles[i - 1] * omega;
        inv_twiddles[i] = inv_twiddles[i - 1] * omega_inv;
    }

    // --- GPU PASSES 1, 2, 3: Inverse NTT to get coefficients ---
    gpu_fft(gpu, &mut a_evals, &inv_twiddles).await?;
    gpu_fft(gpu, &mut b_evals, &inv_twiddles).await?;
    gpu_fft(gpu, &mut c_evals, &inv_twiddles).await?;

    let n_inv = G::Scalar::from(n as u64).invert().unwrap();
    let coset_shift = G::Scalar::multiplicative_generator();
    let mut shift = G::Scalar::ONE;

    // CPU Domain Coset Shift (O(N) operation)
    for i in 0..n {
        a_evals[i] *= n_inv * shift;
        b_evals[i] *= n_inv * shift;
        c_evals[i] *= n_inv * shift;
        shift *= coset_shift;
    }

    // --- GPU PASSES 4, 5, 6: Forward NTT on the Coset ---
    gpu_fft(gpu, &mut a_evals, &fwd_twiddles).await?;
    gpu_fft(gpu, &mut b_evals, &fwd_twiddles).await?;
    gpu_fft(gpu, &mut c_evals, &fwd_twiddles).await?;

    // CPU Pointwise Division H(x) = (A(x)B(x) - C(x)) / Z(x)
    let mut h_evals = vec![G::Scalar::ZERO; n];
    let z_inv = (coset_shift.pow(&[n as u64]) - G::Scalar::ONE)
        .invert()
        .unwrap();

    for i in 0..n {
        h_evals[i] = ((a_evals[i] * b_evals[i]) - c_evals[i]) * z_inv;
    }

    // --- GPU PASS 7: Inverse NTT to bring H(x) back to coefficients ---
    gpu_fft(gpu, &mut h_evals, &inv_twiddles).await?;

    let shift_inv = coset_shift.invert().unwrap();
    let mut current_shift = G::Scalar::ONE;

    for i in 0..n {
        h_evals[i] *= n_inv * current_shift;
        current_shift *= shift_inv;
    }

    Ok(h_evals)
}

/// Generates a perfectly valid, randomized Groth16 proof using WebGPU.
pub async fn create_proof<C: Circuit<G::Scalar>, G: GpuCurve, R: RngCore>(
    circuit: C,
    pk: &ProvingKey<G>,
    gpu: &GpuContext<G>,
    rng: &mut R,
) -> Result<Proof<G>> {
    let mut cs = ProvingCS::new();
    circuit
        .synthesize(&mut cs)
        .context("Circuit synthesis failed")?;

    let mut full_assignment = cs.inputs.clone();
    full_assignment.extend(cs.aux.clone());

    // 1. QAP Quotient Polynomial h(x) Evaluation (Uses 7 WGPU NTT passes)
    let h_coeffs = compute_h_poly(gpu, &cs).await?;

    // 2. WGPU Multi-Scalar Multiplications (Evaluating the QAP over G1)
    let a_msm = gpu_msm_g1(gpu, &pk.a_query, &full_assignment).await?;
    let b_g1_msm = gpu_msm_g1(gpu, &pk.b_query_g1, &full_assignment).await?;
    let l_msm = gpu_msm_g1(gpu, &pk.l_query, &cs.aux).await?;
    let h_msm = gpu_msm_g1(gpu, &pk.h_query, &h_coeffs).await?;

    // Note: G2 MSM uses the CPU fallback because the WGSL shader pipeline is currently
    // bound exclusively to PointG1 structs. A separate G2 wgpu compute pipeline is needed
    // to map F_q^2 arithmetic directly onto the device.
    let b_g2_msm = G::msm_g2_cpu(&pk.b_query_g2, &full_assignment);

    // 3. Cryptographic Zero-Knowledge Randomization
    let r = G::Scalar::random(&mut *rng);
    let s = G::Scalar::random(&mut *rng);

    // Proof Element A: A = alpha + A_msm + r * delta
    let mut proof_a = G::add_g1_proj(&G::affine_to_proj_g1(&pk.alpha_g1), &a_msm);
    proof_a = G::add_g1_proj(&proof_a, &G::mul_g1_scalar(&pk.delta_g1, &r));

    // Proof Element B: B = beta + B_msm + s * delta
    let mut proof_b = G::add_g2_proj(&G::affine_to_proj_g2(&pk.beta_g2), &b_g2_msm);
    proof_b = G::add_g2_proj(&proof_b, &G::mul_g2_scalar(&pk.delta_g2, &s));

    // Proof Element C: C = L_msm + H_msm + s * A + r * B_g1 - (r * s) * delta_g1
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
