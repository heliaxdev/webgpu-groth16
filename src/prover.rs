// TODO: implement a form of efficient batching of GPU operations.
// loading data in and out of the GPU is slow. we want to batch as many ops as possible,
// without using a stupid amount of VRAM (perhaps we need to check
// how much VRAM the host has).

mod constraint_system;
mod gpu_key;
mod h_poly;
mod msm;
mod prepared_key;

pub use gpu_key::{prepare_gpu_proving_key, GpuProvingKey};
pub use h_poly::compute_h_poly;
pub use msm::{gpu_msm_batch, gpu_msm_g1};
pub use prepared_key::{prepare_proving_key, PreparedProvingKey};

use anyhow::Result;
use ff::{Field, PrimeField};
use rand_core::RngCore;

use crate::bellman;
use crate::bucket::{
    compute_bucket_sorting_with_width, compute_glv_bucket_data, compute_glv_bucket_sorting,
    optimal_glv_c,
};
use crate::gpu::curve::GpuCurve;
use crate::gpu::GpuContext;

use constraint_system::GpuConstraintSystem;
use h_poly::{read_h_poly_result, submit_h_poly};
use msm::{
    enqueue_msm_g1, enqueue_msm_g1_persistent, enqueue_msm_g2, enqueue_msm_g2_persistent,
    readback_msms,
};

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

async fn create_proof_with_fixed_randomness<E, G, C>(
    circuit: C,
    pk: &bellman::groth16::Parameters<E>,
    ppk: &PreparedProvingKey<G>,
    gpu: &GpuContext<G>,
    gpu_pk: Option<&GpuProvingKey>,
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
    #[cfg(feature = "timing")]
    let t_phase = std::time::Instant::now();
    let mut cs = GpuConstraintSystem::<G>::new();
    circuit
        .synthesize(&mut cs)
        .map_err(|e| anyhow::anyhow!("circuit synthesis failed: {:?}", e))?;

    // Append input constraints: for each public input i, add the constraint
    // (input[i]) · (1) = (0), which encodes the public input identity.
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
    #[cfg(feature = "timing")]
    eprintln!("[proof] synthesis: {:?} (constraints={num_constraints}, n={n}, inputs={}, aux={})",
        t_phase.elapsed(), cs.inputs.len(), cs.aux.len());

    // Evaluate all linear combinations at the witness to get A, B, C vectors.
    #[cfg(feature = "timing")]
    let t_phase = std::time::Instant::now();
    let mut a_values = vec![G::Scalar::ZERO; n];
    let mut b_values = vec![G::Scalar::ZERO; n];
    let mut c_values = vec![G::Scalar::ZERO; n];

    for i in 0..num_constraints {
        a_values[i] = eval_lc(&cs.a_lcs[i], &cs.inputs, &cs.aux);
        b_values[i] = eval_lc(&cs.b_lcs[i], &cs.inputs, &cs.aux);
        c_values[i] = eval_lc(&cs.c_lcs[i], &cs.inputs, &cs.aux);
    }
    #[cfg(feature = "timing")]
    eprintln!("[proof] eval_lc: {:?}", t_phase.elapsed());

    // Build dense assignments using density masks before submitting H poly,
    // so we can pre-compute GLV bucket data on CPU while GPU processes H.
    #[cfg(feature = "timing")]
    let t_phase = std::time::Instant::now();
    let mut a_assignment = cs.inputs.clone();
    for (i, v) in cs.aux.iter().enumerate() {
        if cs.a_aux_density[i] {
            a_assignment.push(*v);
        }
    }
    let b_assignment =
        dense_assignment_from_masks(&cs.inputs, &cs.aux, &cs.b_input_density, &cs.b_aux_density);
    #[cfg(feature = "timing")]
    eprintln!("[proof] assignments: {:?} (a_assign={}, b_assign={})",
        t_phase.elapsed(), a_assignment.len(), b_assignment.len());

    #[cfg(feature = "timing")]
    let t_phase = std::time::Instant::now();
    // Submit H polynomial to GPU (non-blocking — GPU processes asynchronously).
    let h_pending = submit_h_poly::<G>(gpu, &a_values, &b_values, &c_values)?;
    #[cfg(feature = "timing")]
    eprintln!("[proof] h_poly submit: {:?}", t_phase.elapsed());

    // Pre-compute GLV bucket data for non-H G1 MSMs while GPU computes H.
    // GLV decomposes each scalar k into k1·P + k2·φ(P) with ~128-bit sub-scalars,
    // halving the number of Pippenger windows.
    #[cfg(feature = "timing")]
    let t_phase = std::time::Instant::now();
    // Adaptive bucket width: choose per-MSM c based on point count.
    let a_c = optimal_glv_c(a_assignment.len());
    let b1_c = optimal_glv_c(b_assignment.len());
    let l_c = optimal_glv_c(cs.aux.len());

    // Bucket sorting: with persistent GPU key, GLV negation is folded into sign bits
    // and no combined bases buffer is built. Without it, the original path is used.
    let a_bd;
    let b1_bd;
    let l_bd;
    let b2_bd;
    // Only needed for the non-persistent path:
    let a_glv_bytes;
    let b1_glv_bytes;
    let l_glv_bytes;

    if gpu_pk.is_some() {
        a_bd = compute_glv_bucket_data::<G>(&a_assignment, a_c);
        b1_bd = compute_glv_bucket_data::<G>(&b_assignment, b1_c);
        l_bd = compute_glv_bucket_data::<G>(&cs.aux, l_c);
        b2_bd = compute_bucket_sorting_with_width::<G>(&b_assignment, G::g2_bucket_width());
        a_glv_bytes = Vec::new();
        b1_glv_bytes = Vec::new();
        l_glv_bytes = Vec::new();
    } else {
        let (a_bytes, a_bd_tmp) = compute_glv_bucket_sorting::<G>(
            &a_assignment, &ppk.a_bytes, &ppk.a_phi_bytes, a_c,
        );
        let (b1_bytes, b1_bd_tmp) = compute_glv_bucket_sorting::<G>(
            &b_assignment, &ppk.b_g1_bytes, &ppk.b_g1_phi_bytes, b1_c,
        );
        let (l_bytes, l_bd_tmp) = compute_glv_bucket_sorting::<G>(
            &cs.aux, &ppk.l_bytes, &ppk.l_phi_bytes, l_c,
        );
        a_bd = a_bd_tmp;
        b1_bd = b1_bd_tmp;
        l_bd = l_bd_tmp;
        b2_bd = compute_bucket_sorting_with_width::<G>(&b_assignment, G::g2_bucket_width());
        a_glv_bytes = a_bytes;
        b1_glv_bytes = b1_bytes;
        l_glv_bytes = l_bytes;
    }

    #[cfg(feature = "timing")]
    {
        eprintln!("[proof] bucket sorting (4x GLV): {:?} (c: a={}, b1={}, l={})", t_phase.elapsed(), a_c, b1_c, l_c);
        a_bd.print_distribution_stats("a_g1_glv");
        b1_bd.print_distribution_stats("b1_g1_glv");
        l_bd.print_distribution_stats("l_g1_glv");
        b2_bd.print_distribution_stats("b2_g2");
    }

    #[cfg(feature = "timing")]
    let t_phase = std::time::Instant::now();
    // Await H result (GPU likely already done by now).
    let h_coeffs = read_h_poly_result::<G>(gpu, h_pending).await?;
    #[cfg(feature = "timing")]
    eprintln!("[proof] h_poly read: {:?}", t_phase.elapsed());

    // Enqueue a/b1/l/b2 MSMs right after h_poly completes — GPU starts processing
    // them immediately while CPU computes h bucket sorting below.
    #[cfg(feature = "timing")]
    let t_phase = std::time::Instant::now();
    let (a_job, b1_job, l_job, b2_job);
    if let Some(gpk) = gpu_pk {
        a_job = enqueue_msm_g1_persistent::<G>(gpu, "a", &gpk.a_bases_buf, a_bd)?;
        b1_job = enqueue_msm_g1_persistent::<G>(gpu, "b1", &gpk.b_g1_bases_buf, b1_bd)?;
        l_job = enqueue_msm_g1_persistent::<G>(gpu, "l", &gpk.l_bases_buf, l_bd)?;
        b2_job = enqueue_msm_g2_persistent::<G>(gpu, &gpk.b_g2_bases_buf, b2_bd)?;
    } else {
        a_job = enqueue_msm_g1::<G>(gpu, "a", &a_glv_bytes, a_bd)?;
        b1_job = enqueue_msm_g1::<G>(gpu, "b1", &b1_glv_bytes, b1_bd)?;
        l_job = enqueue_msm_g1::<G>(gpu, "l", &l_glv_bytes, l_bd)?;
        b2_job = enqueue_msm_g2::<G>(gpu, &ppk.b_g2_bytes, b2_bd)?;
    }
    #[cfg(feature = "timing")]
    eprintln!("[proof] msm enqueue a/b1/l/b2: {:?}", t_phase.elapsed());

    // H bucket data depends on h_coeffs — also uses GLV.
    // While CPU computes this, GPU is already processing a/b1/l/b2 MSMs.
    #[cfg(feature = "timing")]
    let t_phase = std::time::Instant::now();
    let h_job;
    let h_c = optimal_glv_c(pk.h.len());
    if let Some(gpk) = gpu_pk {
        let h_bd = compute_glv_bucket_data::<G>(&h_coeffs[..pk.h.len()], h_c);
        #[cfg(feature = "timing")]
        {
            eprintln!("[proof] h bucket sorting (GLV): {:?} (c={})", t_phase.elapsed(), h_c);
            h_bd.print_distribution_stats("h_g1_glv");
        }
        #[cfg(feature = "timing")]
        let t_phase = std::time::Instant::now();
        h_job = enqueue_msm_g1_persistent::<G>(gpu, "h", &gpk.h_bases_buf, h_bd)?;
        #[cfg(feature = "timing")]
        eprintln!("[proof] msm enqueue h: {:?}", t_phase.elapsed());
    } else {
        let (h_glv_bytes, h_bd) = compute_glv_bucket_sorting::<G>(
            &h_coeffs[..pk.h.len()], &ppk.h_bytes, &ppk.h_phi_bytes, h_c,
        );
        #[cfg(feature = "timing")]
        {
            eprintln!("[proof] h bucket sorting (GLV): {:?} (c={})", t_phase.elapsed(), h_c);
            h_bd.print_distribution_stats("h_g1_glv");
        }
        #[cfg(feature = "timing")]
        let t_phase = std::time::Instant::now();
        h_job = enqueue_msm_g1::<G>(gpu, "h", &h_glv_bytes, h_bd)?;
        #[cfg(feature = "timing")]
        eprintln!("[proof] msm enqueue h: {:?}", t_phase.elapsed());
    }

    #[cfg(feature = "timing")]
    let t_phase = std::time::Instant::now();
    let (a_msm, b_g1_msm, l_msm, h_msm, b_g2_msm) =
        readback_msms::<G>(gpu, a_job, b1_job, l_job, h_job, b2_job).await?;
    #[cfg(feature = "timing")]
    eprintln!("[proof] msm readback: {:?}", t_phase.elapsed());

    // Assemble the final Groth16 proof from MSM results and random blinding factors.
    //
    // Groth16 proof elements:
    //   A = α + Σᵢ aᵢ·Aᵢ + r·δ
    //   B = β + Σᵢ bᵢ·Bᵢ + s·δ        (in G2)
    //   C = Σᵢ (aᵢsᵢ)·Lᵢ + h(x)·H + s·A + r·B_G1 − r·s·δ
    #[cfg(feature = "timing")]
    let t_phase = std::time::Instant::now();

    // A = α + a_msm + r·δ
    let mut proof_a = G::add_g1_proj(&G::affine_to_proj_g1(&pk.vk.alpha_g1), &a_msm);
    proof_a = G::add_g1_proj(&proof_a, &G::mul_g1_scalar(&pk.vk.delta_g1, &r));

    // B = β + b_g2_msm + s·δ   (in G2)
    let mut proof_b = G::add_g2_proj(&G::affine_to_proj_g2(&pk.vk.beta_g2), &b_g2_msm);
    proof_b = G::add_g2_proj(&proof_b, &G::mul_g2_scalar(&pk.vk.delta_g2, &s));

    // C = l_msm + h_msm + s·A + r·(β + b_g1_msm + s·δ_G1) − r·s·δ
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
    #[cfg(feature = "timing")]
    eprintln!("[proof] final assembly: {:?}", t_phase.elapsed());

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
    create_proof_with_fixed_randomness::<E, G, C>(circuit, pk, ppk, gpu, None, r, s).await
}

/// Create a Groth16 proof using persistent GPU base buffers.
///
/// Like [`create_proof`] but uses a [`GpuProvingKey`] to skip per-proof base uploads
/// and Montgomery conversion, reusing pre-uploaded GPU buffers across proofs.
pub async fn create_proof_with_gpu_key<E, G, C, R>(
    circuit: C,
    pk: &bellman::groth16::Parameters<E>,
    ppk: &PreparedProvingKey<G>,
    gpu: &GpuContext<G>,
    gpu_pk: &GpuProvingKey,
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
    create_proof_with_fixed_randomness::<E, G, C>(circuit, pk, ppk, gpu, Some(gpu_pk), r, s).await
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

    use super::msm::{fold_window_sums_g1, gpu_msm_g2};
    use super::*;
    use crate::gpu::curve::{G1_GPU_BYTES, G2_GPU_BYTES};

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

        let setup_circuit = DummyCircuit::<Scalar> { x: None, y: None };
        let params =
            bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
                .expect("Failed to generate trusted setup parameters");

        let gpu_ctx = GpuContext::<Bls12>::new()
            .await
            .expect("Failed to initialize WebGPU context");

        let x_value = Scalar::from(3u64);
        let y_value = Scalar::from(27u64);

        let circuit = DummyCircuit {
            x: Some(x_value),
            y: Some(y_value),
        };

        let ppk = prepare_proving_key::<Bls12, Bls12>(&params);
        let proof = create_proof::<Bls12, Bls12, _, _>(circuit, &params, &ppk, &gpu_ctx, &mut rng)
            .await
            .expect("Failed to generate Groth16 proof on GPU");

        let pvk = bellman::groth16::prepare_verifying_key(&params.vk);
        let public_inputs = vec![y_value];
        let is_valid = bellman::groth16::verify_proof(&pvk, &proof, &public_inputs)
            .expect("Failed during proof verification step");
        assert!(is_valid, "The generated Groth16 proof is invalid!");

        // Sanity check: wrong public input should fail
        let wrong_public_inputs = vec![Scalar::from(28u64)];
        let is_valid_wrong = bellman::groth16::verify_proof(&pvk, &proof, &wrong_public_inputs)
            .expect("Failed during proof verification step");
        assert!(
            !is_valid_wrong,
            "The verifier should reject a proof with tampered public inputs"
        );
    }

    #[tokio::test]
    async fn test_gpu_groth16_prover_persistent_key() {
        let mut rng = OsRng;

        let setup_circuit = DummyCircuit::<Scalar> { x: None, y: None };
        let params =
            bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
                .expect("Failed to generate trusted setup parameters");

        let gpu_ctx = GpuContext::<Bls12>::new()
            .await
            .expect("Failed to initialize WebGPU context");

        let ppk = prepare_proving_key::<Bls12, Bls12>(&params);
        let gpu_pk = prepare_gpu_proving_key::<Bls12>(&ppk, &gpu_ctx);

        let x_value = Scalar::from(3u64);
        let y_value = Scalar::from(27u64);

        let circuit = DummyCircuit {
            x: Some(x_value),
            y: Some(y_value),
        };

        let proof = create_proof_with_gpu_key::<Bls12, Bls12, _, _>(
            circuit, &params, &ppk, &gpu_ctx, &gpu_pk, &mut rng,
        )
        .await
        .expect("Failed to generate Groth16 proof with persistent GPU key");

        let pvk = bellman::groth16::prepare_verifying_key(&params.vk);
        let public_inputs = vec![y_value];
        let is_valid = bellman::groth16::verify_proof(&pvk, &proof, &public_inputs)
            .expect("Failed during proof verification step");
        assert!(is_valid, "Proof with persistent GPU key is invalid!");
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

        let t = Instant::now();
        let params = bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup, &mut rng)
            .expect("failed to generate sapling output parameters");
        eprintln!("[diag] param gen: {:?}", t.elapsed());
        eprintln!(
            "[diag] params sizes: a={}, b_g1={}, l={}, h={}, b_g2={}",
            params.a.len(),
            params.b_g1.len(),
            params.l.len(),
            params.h.len(),
            params.b_g2.len()
        );

        let t = Instant::now();
        let gpu_ctx = GpuContext::<Bls12>::new()
            .await
            .expect("failed to initialize gpu");
        eprintln!("[diag] gpu init: {:?}", t.elapsed());

        let circuit = sample_sapling_output_circuit();

        let t = Instant::now();
        let ppk = prepare_proving_key::<Bls12, Bls12>(&params);
        eprintln!("[diag] ppk serialization: {:?}", t.elapsed());

        let t = Instant::now();
        let gpu_pk = prepare_gpu_proving_key::<Bls12>(&ppk, &gpu_ctx);
        eprintln!("[diag] gpu_pk upload+montgomery: {:?}", t.elapsed());

        let t0 = Instant::now();
        let _proof = create_proof_with_gpu_key::<Bls12, Bls12, _, _>(
            circuit, &params, &ppk, &gpu_ctx, &gpu_pk, &mut rng,
        )
        .await
        .expect("gpu sapling output proof failed");
        let dt = t0.elapsed();
        eprintln!("[diag] total proof: {:?}", dt);
    }

    /// eval_lc with an empty linear combination returns zero.
    #[test]
    fn eval_lc_empty() {
        let inputs = vec![Scalar::from(10u64)];
        let aux = vec![Scalar::from(20u64)];
        let lc: Vec<(bellman::Variable, Scalar)> = vec![];
        assert_eq!(eval_lc(&lc, &inputs, &aux), Scalar::ZERO);
    }

    /// eval_lc computes the correct linear combination.
    #[test]
    fn eval_lc_known_value() {
        let inputs = vec![Scalar::from(1u64), Scalar::from(5u64)];
        let aux = vec![Scalar::from(7u64), Scalar::from(11u64)];

        // 3 * input[1] + 2 * aux[0] = 3*5 + 2*7 = 15 + 14 = 29
        let lc = vec![
            (
                bellman::Variable::new_unchecked(bellman::Index::Input(1)),
                Scalar::from(3u64),
            ),
            (
                bellman::Variable::new_unchecked(bellman::Index::Aux(0)),
                Scalar::from(2u64),
            ),
        ];
        let result = eval_lc(&lc, &inputs, &aux);
        assert_eq!(result, Scalar::from(29u64));
    }

    /// dense_assignment_from_masks with all-false masks returns empty.
    #[test]
    fn dense_assignment_all_false() {
        let inputs = vec![Scalar::from(1u64), Scalar::from(2u64)];
        let aux = vec![Scalar::from(3u64), Scalar::from(4u64)];
        let input_mask = vec![false, false];
        let aux_mask = vec![false, false];
        let result = dense_assignment_from_masks(&inputs, &aux, &input_mask, &aux_mask);
        assert!(result.is_empty());
    }

    /// dense_assignment_from_masks with all-true masks returns all values in order.
    #[test]
    fn dense_assignment_all_true() {
        let inputs = vec![Scalar::from(1u64), Scalar::from(2u64)];
        let aux = vec![Scalar::from(3u64), Scalar::from(4u64)];
        let input_mask = vec![true, true];
        let aux_mask = vec![true, true];
        let result = dense_assignment_from_masks(&inputs, &aux, &input_mask, &aux_mask);
        assert_eq!(
            result,
            vec![
                Scalar::from(1u64),
                Scalar::from(2u64),
                Scalar::from(3u64),
                Scalar::from(4u64)
            ]
        );
    }

    /// dense_assignment_from_masks with selective masks picks the right entries.
    #[test]
    fn dense_assignment_selective() {
        let inputs = vec![Scalar::from(10u64), Scalar::from(20u64), Scalar::from(30u64)];
        let aux = vec![Scalar::from(40u64), Scalar::from(50u64)];
        let input_mask = vec![false, true, false];
        let aux_mask = vec![true, false];
        let result = dense_assignment_from_masks(&inputs, &aux, &input_mask, &aux_mask);
        assert_eq!(result, vec![Scalar::from(20u64), Scalar::from(40u64)]);
    }

    /// GpuConstraintSystem starts with one input (the implicit ONE).
    #[test]
    fn gpu_constraint_system_initial_state() {
        let cs = GpuConstraintSystem::<Bls12>::new();
        assert_eq!(cs.inputs.len(), 1);
        assert_eq!(cs.inputs[0], Scalar::ONE);
        assert!(cs.aux.is_empty());
        assert_eq!(cs.b_input_density.len(), 1);
        assert!(!cs.b_input_density[0]);
    }

    /// DummyCircuit synthesizes correctly and produces the right witness.
    #[test]
    fn dummy_circuit_synthesis() {
        let x = Scalar::from(3u64);
        let y = Scalar::from(27u64); // 3^3

        let mut cs = GpuConstraintSystem::<Bls12>::new();
        DummyCircuit {
            x: Some(x),
            y: Some(y),
        }
        .synthesize(&mut cs)
        .expect("synthesis should succeed");

        // Should have 2 inputs: the implicit ONE and y
        assert_eq!(cs.inputs.len(), 2);
        assert_eq!(cs.inputs[1], y);

        // Should have 2 aux: x and x_sq
        assert_eq!(cs.aux.len(), 2);
        assert_eq!(cs.aux[0], x);
        assert_eq!(cs.aux[1], x.square());
    }

    /// DummyCircuit: verify the constraint system is satisfiable.
    #[test]
    fn dummy_circuit_constraints_satisfied() {
        let x = Scalar::from(4u64);
        let y = Scalar::from(64u64); // 4^3

        let mut cs = GpuConstraintSystem::<Bls12>::new();
        DummyCircuit {
            x: Some(x),
            y: Some(y),
        }
        .synthesize(&mut cs)
        .expect("synthesis should succeed");

        // Add input constraints (mirroring create_proof logic)
        for i in 0..cs.inputs.len() {
            cs.a_lcs.push(vec![(
                bellman::Variable::new_unchecked(bellman::Index::Input(i)),
                Scalar::ONE,
            )]);
            cs.b_lcs.push(Vec::new());
            cs.c_lcs.push(Vec::new());
        }

        // Verify A*B = C for each constraint
        for i in 0..cs.a_lcs.len() {
            let a_val = eval_lc(&cs.a_lcs[i], &cs.inputs, &cs.aux);
            let b_val = eval_lc(&cs.b_lcs[i], &cs.inputs, &cs.aux);
            let c_val = eval_lc(&cs.c_lcs[i], &cs.inputs, &cs.aux);
            assert_eq!(
                a_val * b_val,
                c_val,
                "constraint {i} not satisfied: a*b != c"
            );
        }
    }

    /// DummyCircuit with wrong witness (x^3 != y) should fail constraint check.
    #[test]
    fn dummy_circuit_wrong_witness_detected() {
        let x = Scalar::from(3u64);
        let y = Scalar::from(28u64); // Wrong! 3^3 = 27 != 28

        let mut cs = GpuConstraintSystem::<Bls12>::new();
        DummyCircuit {
            x: Some(x),
            y: Some(y),
        }
        .synthesize(&mut cs)
        .expect("synthesis should succeed even with wrong witness");

        // Check that x * x = x_sq holds
        let x_val = cs.aux[0]; // x
        let x_sq = cs.aux[1]; // x_sq
        assert_eq!(x_val * x_val, x_sq, "x * x = x_sq should hold");

        // Check that x_sq * x != y (wrong witness)
        let y_val = cs.inputs[1]; // y
        assert_ne!(x_sq * x_val, y_val, "wrong witness should not satisfy constraint");
    }

    /// fold_window_sums_g1: single identity window should return identity.
    #[test]
    fn fold_window_sums_identity() {
        use crate::gpu::curve::GpuCurve;
        let identity_bytes = <Bls12 as GpuCurve>::serialize_g1(
            &<blstrs::G1Affine as group::prime::PrimeCurveAffine>::identity(),
        );
        let result = fold_window_sums_g1::<Bls12>(&identity_bytes, 1, 13).unwrap();
        assert!(
            bool::from(result.is_identity()),
            "folding a single identity window should give identity"
        );
    }

    /// fold_window_sums_g1 with one non-trivial window returns that point.
    #[test]
    fn fold_window_sums_single_point() {
        use crate::gpu::curve::GpuCurve;
        let g = <blstrs::G1Affine as group::prime::PrimeCurveAffine>::generator();
        let g_bytes = <Bls12 as GpuCurve>::serialize_g1(&g);
        let result = fold_window_sums_g1::<Bls12>(&g_bytes, 1, 13).unwrap();
        let result_affine: blstrs::G1Affine = result.into();
        assert_eq!(
            result_affine, g,
            "single window fold should return the window sum"
        );
    }

    /// fold_window_sums_g1: two windows, verify Horner-style folding.
    /// result = window[1] * 2^c + window[0]
    #[test]
    fn fold_window_sums_two_windows() {
        use crate::gpu::curve::GpuCurve;
        use group::prime::PrimeCurveAffine;

        let g = blstrs::G1Affine::generator();
        let g_proj = blstrs::G1Projective::from(g);
        let c = 13usize;

        // window 0 = G, window 1 = 2G
        let p0 = g;
        let p1: blstrs::G1Affine = (g_proj + g_proj).into();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&<Bls12 as GpuCurve>::serialize_g1(&p0));
        bytes.extend_from_slice(&<Bls12 as GpuCurve>::serialize_g1(&p1));

        let result = fold_window_sums_g1::<Bls12>(&bytes, 2, c).unwrap();

        // Expected: start with window[1]=2G, double c times, add window[0]=G
        // = 2G * 2^13 + G = G * (2^14 + 1) = G * 16385
        let expected = g_proj * Scalar::from(16385u64);
        assert_eq!(
            blstrs::G1Affine::from(result),
            blstrs::G1Affine::from(expected),
            "two-window fold mismatch"
        );
    }

    /// CPU proof generation and verification across several (x, y=x^3) pairs.
    #[test]
    fn cpu_proof_multiple_witnesses() {
        let mut rng = OsRng;
        let setup_circuit = DummyCircuit::<Scalar> { x: None, y: None };
        let params =
            bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
                .expect("param gen failed");
        let pvk = bellman::groth16::prepare_verifying_key(&params.vk);

        for x_val in [1u64, 2, 3, 5, 100, 999] {
            let x = Scalar::from(x_val);
            let y = x * x * x;
            let circuit = DummyCircuit {
                x: Some(x),
                y: Some(y),
            };

            let proof = bellman::groth16::create_random_proof(circuit, &params, &mut rng)
                .expect("cpu proof failed");
            let valid = bellman::groth16::verify_proof(&pvk, &proof, &[y])
                .expect("verification failed");
            assert!(valid, "proof for x={x_val} should verify");
        }
    }

    /// A CPU proof with incorrect witness should fail verification.
    #[test]
    fn cpu_proof_wrong_public_input_rejects() {
        let mut rng = OsRng;
        let setup_circuit = DummyCircuit::<Scalar> { x: None, y: None };
        let params =
            bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
                .expect("param gen failed");
        let pvk = bellman::groth16::prepare_verifying_key(&params.vk);

        let x = Scalar::from(5u64);
        let y = x * x * x; // 125
        let circuit = DummyCircuit {
            x: Some(x),
            y: Some(y),
        };

        let proof = bellman::groth16::create_random_proof(circuit, &params, &mut rng)
            .expect("cpu proof failed");

        // Should verify with correct y
        let valid = bellman::groth16::verify_proof(&pvk, &proof, &[y]).expect("verify failed");
        assert!(valid);

        // Should reject with wrong y
        let wrong_y = Scalar::from(126u64);
        let invalid = bellman::groth16::verify_proof(&pvk, &proof, &[wrong_y]).expect("verify failed");
        assert!(!invalid, "should reject proof with wrong public input");
    }

    /// PreparedProvingKey serialization produces expected byte lengths.
    #[test]
    fn prepared_proving_key_sizes() {
        let mut rng = OsRng;
        let setup_circuit = DummyCircuit::<Scalar> { x: None, y: None };
        let params =
            bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
                .expect("param gen failed");

        let ppk = prepare_proving_key::<Bls12, Bls12>(&params);

        assert_eq!(ppk.a_bytes.len(), params.a.len() * G1_GPU_BYTES);
        assert_eq!(ppk.b_g1_bytes.len(), params.b_g1.len() * G1_GPU_BYTES);
        assert_eq!(ppk.l_bytes.len(), params.l.len() * G1_GPU_BYTES);
        assert_eq!(ppk.h_bytes.len(), params.h.len() * G1_GPU_BYTES);
        assert_eq!(ppk.b_g2_bytes.len(), params.b_g2.len() * G2_GPU_BYTES);
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
    async fn test_gpu_msm_scalar2_matches_cpu() {
        let mut rng = OsRng;
        let setup_circuit = DummyCircuit::<Scalar> { x: None, y: None };
        let params =
            bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
                .expect("failed to generate parameters");
        let gpu_ctx = GpuContext::<Bls12>::new()
            .await
            .expect("failed to initialize gpu");

        let base = params.a[0];

        // scalar = 2: exercises double_g1 via scalar_mul_g1
        let gpu = gpu_msm_g1::<Bls12>(&gpu_ctx, &[base], &[Scalar::from(2u64)])
            .await
            .expect("gpu msm scalar=2 failed");
        let cpu: blstrs::G1Projective = base.into();
        let cpu_2 = cpu + cpu;
        assert_eq!(
            Bls12::proj_to_affine_g1(&gpu),
            Bls12::proj_to_affine_g1(&cpu_2),
            "scalar=2 mismatch"
        );

        // scalar = 3: exercises double_g1 + add_g1 via scalar_mul_g1
        let gpu = gpu_msm_g1::<Bls12>(&gpu_ctx, &[base], &[Scalar::from(3u64)])
            .await
            .expect("gpu msm scalar=3 failed");
        let cpu_3 = cpu_2 + cpu;
        assert_eq!(
            Bls12::proj_to_affine_g1(&gpu),
            Bls12::proj_to_affine_g1(&cpu_3),
            "scalar=3 mismatch"
        );

        // 2 points, scalars [1, 1]: exercises add_g1_mixed in aggregate_buckets
        let base2 = params.a[1];
        let gpu = gpu_msm_g1::<Bls12>(&gpu_ctx, &[base, base2], &[Scalar::from(1u64), Scalar::from(1u64)])
            .await
            .expect("gpu msm 2-point failed");
        let cpu_sum: blstrs::G1Projective = Into::<blstrs::G1Projective>::into(base) + Into::<blstrs::G1Projective>::into(base2);
        assert_eq!(
            Bls12::proj_to_affine_g1(&gpu),
            Bls12::proj_to_affine_g1(&cpu_sum),
            "2-point [1,1] mismatch"
        );
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
            None,
            r,
            s,
        )
        .await
        .expect("gpu proof creation failed");

        let cpu_proof = bellman::groth16::create_proof(
            DummyCircuit {
                x: Some(x_value),
                y: Some(y_value),
            },
            &params,
            r,
            s,
        )
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
