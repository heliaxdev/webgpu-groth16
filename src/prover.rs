//! Groth16 proof construction using GPU-accelerated MSM and NTT.
//!
//! The main entry point is [`create_proof`].
//!
//! Proof construction flow:
//! 1. Circuit synthesis → constraint system (A, B, C linear combinations)
//! 2. Witness evaluation → dense A/B/C coefficient vectors
//! 3. H-polynomial: `H(x) = (A(x)·B(x) − C(x)) / Z(x)` via GPU NTT pipeline
//! 4. Five MSMs dispatched to GPU: `a` (G1), `b1` (G1), `l` (G1), `h` (G1),
//!    `b2` (G2)
//! 5. CPU-side proof assembly with random blinding factors r, s

mod constraint_system;
pub(crate) mod density_masks;
mod gpu_key;
mod h_poly;
mod msm;
mod prepared_key;

use anyhow::Result;
use ff::{Field, PrimeField};
use rand_core::RngCore;

use self::constraint_system::GpuConstraintSystem;
use self::density_masks::dense_assignment_from_masks;
pub use self::gpu_key::{GpuProvingKey, prepare_gpu_proving_key};
pub use self::h_poly::compute_h_poly;
use self::h_poly::{read_h_poly_result, submit_h_poly};
use self::msm::{MsmBases, enqueue_msm, readback_msms};
pub use self::msm::{gpu_msm_batch, gpu_msm_g1};
pub use self::prepared_key::{PreparedProvingKey, prepare_proving_key};
use crate::bellman;
use crate::bucket::{
    compute_bucket_sorting_with_width, compute_glv_bucket_data,
    compute_glv_bucket_sorting, optimal_glv_c,
};
use crate::gpu::GpuContext;
use crate::gpu::curve::GpuCurve;

/// Proving key required to create a new Groth16 proof with [`create_proof`].
#[derive(Copy, Clone)]
pub enum ProvingKey<'key, G: GpuCurve> {
    /// Use a key that has already been uploaded to the GPU.
    Uploaded(&'key GpuProvingKey<G>),
    /// Use a key that has yet to be uploaded to the GPU.
    Serialized(&'key PreparedProvingKey<G>),
}

fn marshal_scalars<G: GpuCurve>(scalars: &[G::Scalar]) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(scalars.len() * 32);
    for s in scalars {
        buffer.extend_from_slice(&G::serialize_scalar(s));
    }
    buffer
}

fn eval_lc<S: PrimeField>(
    lc: &[(bellman::Variable, S)],
    inputs: &[S],
    aux: &[S],
) -> S {
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

/// Core proof construction with fixed randomness (deterministic for testing).
///
/// Orchestrates the full Groth16 proving pipeline:
/// 1. Synthesize the circuit into a constraint system
/// 2. Submit H-polynomial computation to GPU (non-blocking)
/// 3. Compute GLV bucket sorting on CPU (overlapped with GPU H-poly work)
/// 4. Enqueue 5 MSMs (a, b1, l, b2, then h after H-poly completes)
/// 5. Read back MSM results and assemble the final proof (A, B, C)
async fn create_proof_with_fixed_randomness<E, G, C>(
    circuit: C,
    pk: ProvingKey<'_, G>,
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
    eprintln!(
        "[proof] synthesis: {:?} (constraints={num_constraints}, n={n}, \
         inputs={}, aux={})",
        t_phase.elapsed(),
        cs.inputs.len(),
        cs.aux.len()
    );

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
        if cs.a_aux_density.is_set(i) {
            a_assignment.push(*v);
        }
    }
    let b_assignment = dense_assignment_from_masks(
        &cs.inputs,
        &cs.aux,
        &cs.b_input_density,
        &cs.b_aux_density,
    );
    #[cfg(feature = "timing")]
    eprintln!(
        "[proof] assignments: {:?} (a_assign={}, b_assign={})",
        t_phase.elapsed(),
        a_assignment.len(),
        b_assignment.len()
    );

    #[cfg(feature = "timing")]
    let t_phase = std::time::Instant::now();
    // Submit H polynomial to GPU (non-blocking — GPU processes asynchronously).
    let h_pending = submit_h_poly::<G>(gpu, &a_values, &b_values, &c_values)?;
    #[cfg(feature = "timing")]
    eprintln!("[proof] h_poly submit: {:?}", t_phase.elapsed());

    // Pre-compute GLV bucket data for non-H G1 MSMs while GPU computes H.
    // GLV decomposes each scalar k into k1·P + k2·φ(P) with ~128-bit
    // sub-scalars, halving the number of Pippenger windows.
    #[cfg(feature = "timing")]
    let t_phase = std::time::Instant::now();
    // Adaptive bucket width: choose per-MSM c based on point count.
    let a_c = optimal_glv_c::<G>(a_assignment.len());
    let b1_c = optimal_glv_c::<G>(b_assignment.len());
    let l_c = optimal_glv_c::<G>(cs.aux.len());

    // Bucket sorting: with persistent GPU key, GLV negation is folded into sign
    // bits and no combined bases buffer is built. Without it, the original
    // path is used.
    let a_bd;
    let b1_bd;
    let l_bd;
    let b2_bd;
    // Only needed for the non-persistent path:
    let a_glv_bytes;
    let b1_glv_bytes;
    let l_glv_bytes;
    match pk {
        ProvingKey::Uploaded(_) => {
            a_bd = compute_glv_bucket_data::<G>(&a_assignment, a_c);
            b1_bd = compute_glv_bucket_data::<G>(&b_assignment, b1_c);
            l_bd = compute_glv_bucket_data::<G>(&cs.aux, l_c);
            b2_bd = compute_bucket_sorting_with_width::<G>(
                &b_assignment,
                G::g2_bucket_width(),
            );
            a_glv_bytes = Vec::new();
            b1_glv_bytes = Vec::new();
            l_glv_bytes = Vec::new();
        }
        ProvingKey::Serialized(ppk) => {
            let (a_bytes, a_bd_tmp) = compute_glv_bucket_sorting::<G>(
                &a_assignment,
                &ppk.a_bytes,
                ppk.a_phi_bytes.as_deref().unwrap_or(&[]),
                a_c,
            );
            let (b1_bytes, b1_bd_tmp) = compute_glv_bucket_sorting::<G>(
                &b_assignment,
                &ppk.b_g1_bytes,
                ppk.b_g1_phi_bytes.as_deref().unwrap_or(&[]),
                b1_c,
            );
            let (l_bytes, l_bd_tmp) = compute_glv_bucket_sorting::<G>(
                &cs.aux,
                &ppk.l_bytes,
                ppk.l_phi_bytes.as_deref().unwrap_or(&[]),
                l_c,
            );
            a_bd = a_bd_tmp;
            b1_bd = b1_bd_tmp;
            l_bd = l_bd_tmp;
            b2_bd = compute_bucket_sorting_with_width::<G>(
                &b_assignment,
                G::g2_bucket_width(),
            );
            a_glv_bytes = a_bytes;
            b1_glv_bytes = b1_bytes;
            l_glv_bytes = l_bytes;
        }
    }

    #[cfg(feature = "timing")]
    {
        eprintln!(
            "[proof] bucket sorting (4x GLV): {:?} (c: a={}, b1={}, l={})",
            t_phase.elapsed(),
            a_c,
            b1_c,
            l_c
        );
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

    // Enqueue a/b1/l/b2 MSMs right after h_poly completes — GPU starts
    // processing them immediately while CPU computes h bucket sorting
    // below.
    #[cfg(feature = "timing")]
    let t_phase = std::time::Instant::now();
    let (a_job, b1_job, l_job, b2_job);
    match pk {
        ProvingKey::Uploaded(gpk) => {
            a_job = enqueue_msm::<G>(
                gpu,
                "a",
                MsmBases::Persistent(&gpk.a_bases_buf),
                a_bd,
                false,
            )?;
            b1_job = enqueue_msm::<G>(
                gpu,
                "b1",
                MsmBases::Persistent(&gpk.b_g1_bases_buf),
                b1_bd,
                false,
            )?;
            l_job = enqueue_msm::<G>(
                gpu,
                "l",
                MsmBases::Persistent(&gpk.l_bases_buf),
                l_bd,
                false,
            )?;
            b2_job = enqueue_msm::<G>(
                gpu,
                "b2",
                MsmBases::Persistent(&gpk.b_g2_bases_buf),
                b2_bd,
                true,
            )?;
        }
        ProvingKey::Serialized(ppk) => {
            a_job = enqueue_msm::<G>(
                gpu,
                "a",
                MsmBases::Bytes(&a_glv_bytes),
                a_bd,
                false,
            )?;
            b1_job = enqueue_msm::<G>(
                gpu,
                "b1",
                MsmBases::Bytes(&b1_glv_bytes),
                b1_bd,
                false,
            )?;
            l_job = enqueue_msm::<G>(
                gpu,
                "l",
                MsmBases::Bytes(&l_glv_bytes),
                l_bd,
                false,
            )?;
            b2_job = enqueue_msm::<G>(
                gpu,
                "b2",
                MsmBases::Bytes(&ppk.b_g2_bytes),
                b2_bd,
                true,
            )?;
        }
    }
    #[cfg(feature = "timing")]
    eprintln!("[proof] msm enqueue a/b1/l/b2: {:?}", t_phase.elapsed());

    // H bucket data depends on h_coeffs — also uses GLV.
    // While CPU computes this, GPU is already processing a/b1/l/b2 MSMs.
    #[cfg(feature = "timing")]
    let t_phase = std::time::Instant::now();
    let h_job = match pk {
        ProvingKey::Uploaded(gpu_pk) => {
            let h_c = optimal_glv_c::<G>(gpu_pk.h_len);
            let h_bd =
                compute_glv_bucket_data::<G>(&h_coeffs[..gpu_pk.h_len], h_c);
            #[cfg(feature = "timing")]
            {
                eprintln!(
                    "[proof] h bucket sorting (GLV): {:?} (c={})",
                    t_phase.elapsed(),
                    h_c
                );
                h_bd.print_distribution_stats("h_g1_glv");
            }
            #[cfg(feature = "timing")]
            let t_phase = std::time::Instant::now();
            let h_job = enqueue_msm::<G>(
                gpu,
                "h",
                MsmBases::Persistent(&gpu_pk.h_bases_buf),
                h_bd,
                false,
            )?;
            #[cfg(feature = "timing")]
            eprintln!("[proof] msm enqueue h: {:?}", t_phase.elapsed());
            h_job
        }
        ProvingKey::Serialized(ppk) => {
            let h_c = optimal_glv_c::<G>(ppk.h_len);
            let (h_glv_bytes, h_bd) = compute_glv_bucket_sorting::<G>(
                &h_coeffs[..ppk.h_len],
                &ppk.h_bytes,
                ppk.h_phi_bytes.as_deref().unwrap_or(&[]),
                h_c,
            );
            #[cfg(feature = "timing")]
            {
                eprintln!(
                    "[proof] h bucket sorting (GLV): {:?} (c={})",
                    t_phase.elapsed(),
                    h_c
                );
                h_bd.print_distribution_stats("h_g1_glv");
            }
            #[cfg(feature = "timing")]
            let t_phase = std::time::Instant::now();
            let h_job = enqueue_msm::<G>(
                gpu,
                "h",
                MsmBases::Bytes(&h_glv_bytes),
                h_bd,
                false,
            )?;
            #[cfg(feature = "timing")]
            eprintln!("[proof] msm enqueue h: {:?}", t_phase.elapsed());
            h_job
        }
    };

    #[cfg(feature = "timing")]
    let t_phase = std::time::Instant::now();
    let (a_msm, b_g1_msm, l_msm, h_msm, b_g2_msm) =
        readback_msms::<G>(gpu, a_job, b1_job, l_job, h_job, b2_job).await?;
    #[cfg(feature = "timing")]
    eprintln!("[proof] msm readback: {:?}", t_phase.elapsed());

    // Assemble the final Groth16 proof from MSM results and random blinding
    // factors.
    //
    // Groth16 proof elements:
    //   A = α + Σᵢ aᵢ·Aᵢ + r·δ
    //   B = β + Σᵢ bᵢ·Bᵢ + s·δ        (in G2)
    //   C = Σᵢ (aᵢsᵢ)·Lᵢ + h(x)·H + s·A + r·B_G1 − r·s·δ
    #[cfg(feature = "timing")]
    let t_phase = std::time::Instant::now();

    let (alpha_g1, beta_g1, beta_g2, delta_g1, delta_g2) = match pk {
        ProvingKey::Uploaded(k) => (
            &k.alpha_g1,
            &k.beta_g1,
            &k.beta_g2,
            &k.delta_g1,
            &k.delta_g2,
        ),
        ProvingKey::Serialized(k) => (
            &k.alpha_g1,
            &k.beta_g1,
            &k.beta_g2,
            &k.delta_g1,
            &k.delta_g2,
        ),
    };

    // A = α + a_msm + r·δ
    let mut proof_a = G::add_g1_proj(&G::affine_to_proj_g1(alpha_g1), &a_msm);
    proof_a = G::add_g1_proj(&proof_a, &G::mul_g1_scalar(delta_g1, &r));

    // B = β + b_g2_msm + s·δ   (in G2)
    let mut proof_b = G::add_g2_proj(&G::affine_to_proj_g2(beta_g2), &b_g2_msm);
    proof_b = G::add_g2_proj(&proof_b, &G::mul_g2_scalar(delta_g2, &s));

    // C = l_msm + h_msm + s·A + r·(β + b_g1_msm + s·δ_G1) − r·s·δ
    let mut proof_c = G::add_g1_proj(&l_msm, &h_msm);
    let mut b_g1 = G::add_g1_proj(&G::affine_to_proj_g1(beta_g1), &b_g1_msm);
    b_g1 = G::add_g1_proj(&b_g1, &G::mul_g1_scalar(delta_g1, &s));

    let c_shift_a = G::mul_g1_proj_scalar(&proof_a, &s);
    proof_c = G::add_g1_proj(&proof_c, &c_shift_a);

    let c_shift_b = G::mul_g1_proj_scalar(&b_g1, &r);
    proof_c = G::add_g1_proj(&proof_c, &c_shift_b);

    let mut rs = r;
    rs *= s;
    let rs_delta = G::mul_g1_scalar(delta_g1, &rs);
    proof_c = G::sub_g1_proj(&proof_c, &rs_delta);
    #[cfg(feature = "timing")]
    eprintln!("[proof] final assembly: {:?}", t_phase.elapsed());

    Ok(bellman::groth16::Proof {
        a: G::proj_to_affine_g1(&proof_a),
        b: G::proj_to_affine_g2(&proof_b),
        c: G::proj_to_affine_g1(&proof_c),
    })
}

/// Create a new Groth16 proof.
///
/// Uses a [`GpuProvingKey`] to skip per-proof base uploads and Montgomery
/// conversion, reusing pre-uploaded GPU buffers across proofs.
pub async fn create_proof<E, G, C, R>(
    circuit: C,
    pk: ProvingKey<'_, G>,
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

    create_proof_with_fixed_randomness::<E, G, C>(circuit, pk, gpu, r, s).await
}

#[cfg(test)]
mod tests;
