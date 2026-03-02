use crate::bellman::Circuit;
use blstrs::{Bls12, Scalar};
use ff::Field;
use group::{Curve, Group};
use masp_primitives::asset_type::AssetType;
use masp_primitives::jubjub;
use masp_primitives::sapling::Note;
use masp_primitives::sapling::PaymentAddress;
use masp_primitives::sapling::Rseed;
use masp_primitives::sapling::{Diversifier, ProofGenerationKey};
use masp_proofs::circuit::sapling::Output as SaplingOutputCircuit;
use rand_core::OsRng;
use std::time::Instant;

use super::msm::{fold_window_sums_g1, gpu_msm_g2};
use super::*;
use crate::gpu::curve::GpuCurve;

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
    let is_valid =
        bellman::groth16::verify_proof(&pvk, &proof, &public_inputs).expect("verification failed");
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

fn sample_sapling_output_note() -> (PaymentAddress, Note) {
    let asset_type = AssetType::new(b"benchmark-asset").expect("asset type creation failed");

    let pgk = ProofGenerationKey {
        ak: jubjub::SubgroupPoint::generator(),
        nsk: jubjub::Fr::from(11u64),
    };
    let vk = pgk.to_viewing_key();
    let mut payment_address = None;
    for d0 in 0u8..=255 {
        if let Some(addr) = vk.to_payment_address(Diversifier([d0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) {
            payment_address = Some(addr);
            break;
        }
    }
    let payment_address = payment_address.expect("failed to find a valid diversifier");

    (
        payment_address,
        Note {
            asset_type,
            value: 42,
            g_d: payment_address.g_d().unwrap(),
            pk_d: *payment_address.pk_d(),
            rseed: Rseed::AfterZip212([0xaf; 32]),
        },
    )
}

fn sample_sapling_output_circuit() -> SaplingOutputCircuit {
    let (pa, note) = sample_sapling_output_note();
    let value_commitment = note
        .asset_type
        .value_commitment(note.value, jubjub::Fr::from(7u64));

    let rcm = note.rcm();

    SaplingOutputCircuit {
        value_commitment: Some(value_commitment),
        asset_identifier: note.asset_type.identifier_bits(),
        payment_address: Some(pa),
        commitment_randomness: Some(rcm),
        esk: Some(note.derive_esk().unwrap()),
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
    eprintln!("CPU Sapling Output proof took {:?}", dt);

    // verify proof
    let inputs = {
        let (_, note) = sample_sapling_output_note();
        let esk = note.derive_esk().unwrap();
        let epk: jubjub::AffinePoint = jubjub::ExtendedPoint::from(note.g_d * esk).into();
        let cv: jubjub::AffinePoint = jubjub::ExtendedPoint::from(
            note.asset_type
                .value_commitment(note.value, jubjub::Fr::from(7u64))
                .commitment(),
        )
        .into();
        let cmu = note.cmu();
        [cv.get_u(), cv.get_v(), epk.get_u(), epk.get_v(), cmu]
    };
    let valid = bellman::groth16::verify_proof(
        &bellman::groth16::prepare_verifying_key(&params.vk),
        &proof,
        &inputs,
    )
    .expect("verification failed");
    assert!(valid, "output proof should be valid");
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
    let proof = create_proof_with_gpu_key::<Bls12, Bls12, _, _>(
        circuit, &params, &ppk, &gpu_ctx, &gpu_pk, &mut rng,
    )
    .await
    .expect("gpu sapling output proof failed");
    let dt = t0.elapsed();
    eprintln!("[diag] total proof: {:?}", dt);

    // verify proof
    let inputs = {
        let (_, note) = sample_sapling_output_note();
        let esk = note.derive_esk().unwrap();
        let epk: jubjub::AffinePoint = jubjub::ExtendedPoint::from(note.g_d * esk).into();
        let cv: jubjub::AffinePoint = jubjub::ExtendedPoint::from(
            note.asset_type
                .value_commitment(note.value, jubjub::Fr::from(7u64))
                .commitment(),
        )
        .into();
        let cmu = note.cmu();
        [cv.get_u(), cv.get_v(), epk.get_u(), epk.get_v(), cmu]
    };
    let valid = bellman::groth16::verify_proof(
        &bellman::groth16::prepare_verifying_key(&params.vk),
        &proof,
        &inputs,
    )
    .expect("verification failed");
    assert!(valid, "output proof should be valid");
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
    let inputs = vec![
        Scalar::from(10u64),
        Scalar::from(20u64),
        Scalar::from(30u64),
    ];
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
    assert_ne!(
        x_sq * x_val,
        y_val,
        "wrong witness should not satisfy constraint"
    );
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
        let valid =
            bellman::groth16::verify_proof(&pvk, &proof, &[y]).expect("verification failed");
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

    assert_eq!(
        ppk.a_bytes.len(),
        params.a.len() * <Bls12 as GpuCurve>::G1_GPU_BYTES
    );
    assert_eq!(
        ppk.b_g1_bytes.len(),
        params.b_g1.len() * <Bls12 as GpuCurve>::G1_GPU_BYTES
    );
    assert_eq!(
        ppk.l_bytes.len(),
        params.l.len() * <Bls12 as GpuCurve>::G1_GPU_BYTES
    );
    assert_eq!(
        ppk.h_bytes.len(),
        params.h.len() * <Bls12 as GpuCurve>::G1_GPU_BYTES
    );
    assert_eq!(
        ppk.b_g2_bytes.len(),
        params.b_g2.len() * <Bls12 as GpuCurve>::G2_GPU_BYTES
    );
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
    let gpu = gpu_msm_g1::<Bls12>(
        &gpu_ctx,
        &[base, base2],
        &[Scalar::from(1u64), Scalar::from(1u64)],
    )
    .await
    .expect("gpu msm 2-point failed");
    let cpu_sum: blstrs::G1Projective =
        Into::<blstrs::G1Projective>::into(base) + Into::<blstrs::G1Projective>::into(base2);
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
    let b_assignment =
        dense_assignment_from_masks(&cs.inputs, &cs.aux, &cs.b_input_density, &cs.b_aux_density);

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

#[tokio::test]
async fn test_gpu_msm_30k_matches_cpu() {
    let mut rng = OsRng;
    let n = 30_000;

    // Generate 30k random points and scalars
    let points: Vec<_> = (0..n)
        .map(|_| <Bls12 as pairing::Engine>::G1::random(&mut rng).to_affine())
        .collect();
    let scalars: Vec<_> = (0..n).map(|_| Scalar::random(&mut rng)).collect();

    let gpu_ctx = GpuContext::<Bls12>::new().await.unwrap();
    let gpu_res = gpu_msm_g1::<Bls12>(&gpu_ctx, &points, &scalars)
        .await
        .unwrap();

    let mut cpu_res = <Bls12 as pairing::Engine>::G1::identity();
    for (p, s) in points.iter().zip(scalars.iter()) {
        cpu_res += *p * *s;
    }

    assert_eq!(
        Bls12::proj_to_affine_g1(&gpu_res),
        Bls12::proj_to_affine_g1(&cpu_res),
        "GPU MSM failed at N=30,000"
    );
}

#[derive(Clone)]
struct LargeDummyCircuit<Scalar: PrimeField> {
    pub x: Option<Scalar>,
    pub y: Option<Scalar>,
    pub num_constraints: usize,
}

impl<Scalar: PrimeField> bellman::Circuit<Scalar> for LargeDummyCircuit<Scalar> {
    fn synthesize<CS: bellman::ConstraintSystem<Scalar>>(
        self,
        cs: &mut CS,
    ) -> Result<(), bellman::SynthesisError> {
        let x_val = self.x;
        let y_val = self.y;

        let y = cs.alloc_input(
            || "y",
            || y_val.ok_or(bellman::SynthesisError::AssignmentMissing),
        )?;
        let x = cs.alloc(
            || "x",
            || x_val.ok_or(bellman::SynthesisError::AssignmentMissing),
        )?;

        let mut curr = x;
        let mut curr_val = x_val;

        // Add tens of thousands of constraints: curr * 2 = next
        for i in 0..self.num_constraints {
            let next_val = curr_val.map(|v| v + v);
            let next = cs.alloc(
                || format!("next_{i}"),
                || next_val.ok_or(bellman::SynthesisError::AssignmentMissing),
            )?;

            // Enforce: curr * 2 = next  ==> curr * 2 = next
            cs.enforce(
                || format!("c_{i}"),
                |lc| lc + curr,
                |lc| lc + CS::one() + CS::one(),
                |lc| lc + next,
            );

            curr = next;
            curr_val = next_val;
        }

        // Finally tie it to the public input y: curr * 1 = y
        cs.enforce(|| "tie_y", |lc| lc + curr, |lc| lc + CS::one(), |lc| lc + y);
        Ok(())
    }
}

#[tokio::test]
async fn test_gpu_large_circuit_abc_match() {
    let mut rng = OsRng;
    let num_constraints = 32700; // Pushes FFT to N=32768

    let setup_circuit = LargeDummyCircuit::<Scalar> {
        x: None,
        y: None,
        num_constraints,
    };
    let params =
        bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
            .expect("param gen failed");

    let gpu_ctx = GpuContext::<Bls12>::new().await.unwrap();
    let ppk = prepare_proving_key::<Bls12, Bls12>(&params);
    let gpu_pk = prepare_gpu_proving_key::<Bls12>(&ppk, &gpu_ctx);

    // If x = 1, multiplying by 2 `num_constraints` times means y = 1 * 2^(num_constraints)
    let x_value = Scalar::ONE;
    let mut y_value = x_value;
    for _ in 0..num_constraints {
        y_value = y_value + y_value;
    }

    let circuit = LargeDummyCircuit {
        x: Some(x_value),
        y: Some(y_value),
        num_constraints,
    };

    // Fix randomness to compare components exactly
    let r = Scalar::ZERO;
    let s = Scalar::ZERO;

    let t0 = Instant::now();
    let gpu_proof = create_proof_with_fixed_randomness::<Bls12, Bls12, _>(
        circuit.clone(),
        &params,
        &ppk,
        &gpu_ctx,
        Some(&gpu_pk),
        r,
        s,
    )
    .await
    .expect("gpu proof failed");
    eprintln!("GPU Large Proof took: {:?}", t0.elapsed());

    let cpu_proof =
        bellman::groth16::create_proof(circuit, &params, r, s).expect("cpu proof failed");

    let gpu_a: blstrs::G1Projective = gpu_proof.a.into();
    let cpu_a: blstrs::G1Projective = cpu_proof.a.into();
    let gpu_b: blstrs::G2Projective = gpu_proof.b.into();
    let cpu_b: blstrs::G2Projective = cpu_proof.b.into();
    let gpu_c: blstrs::G1Projective = gpu_proof.c.into();
    let cpu_c: blstrs::G1Projective = cpu_proof.c.into();

    assert_eq!(
        Bls12::proj_to_affine_g1(&gpu_a),
        Bls12::proj_to_affine_g1(&cpu_a),
        "A component mismatch (Density filtering issue?)"
    );
    assert_eq!(
        Bls12::proj_to_affine_g2(&gpu_b),
        Bls12::proj_to_affine_g2(&cpu_b),
        "B component mismatch (Density filtering issue?)"
    );
    assert_eq!(
        Bls12::proj_to_affine_g1(&gpu_c),
        Bls12::proj_to_affine_g1(&cpu_c),
        "C component mismatch (FFT / H-poly issue!)"
    );
}
