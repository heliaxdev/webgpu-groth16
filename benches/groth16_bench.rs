use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use blstrs::{Bls12, Scalar};
use ff::Field;
use group::{Curve, Group};
use masp_primitives::asset_type::AssetType;
use masp_primitives::jubjub;
use masp_primitives::sapling::{Diversifier, ProofGenerationKey};
use masp_proofs::circuit::sapling::Output as SaplingOutputCircuit;
use rand_core::OsRng;

use webgpu_groth16::bellman;
use webgpu_groth16::bucket::compute_bucket_sorting;
use webgpu_groth16::gpu::GpuContext;
use webgpu_groth16::gpu::curve::GpuCurve;
use webgpu_groth16::prover;
use webgpu_groth16::prover::PreparedProvingKey;

// ---------------------------------------------------------------------------
// Repeated-squaring circuit: x -> x^2 -> x^4 -> ... -> x^(2^num_squarings)
// Produces exactly `num_squarings` R1CS constraints.
// ---------------------------------------------------------------------------
struct RepeatedSquaringCircuit<S: ff::PrimeField> {
    x: Option<S>,
    num_squarings: usize,
}

impl<S: ff::PrimeField> bellman::Circuit<S> for RepeatedSquaringCircuit<S> {
    fn synthesize<CS: bellman::ConstraintSystem<S>>(
        self,
        cs: &mut CS,
    ) -> Result<(), bellman::SynthesisError> {
        let mut cur_val = self.x;
        let mut cur_var = cs.alloc(
            || "x",
            || cur_val.ok_or(bellman::SynthesisError::AssignmentMissing),
        )?;

        for i in 0..self.num_squarings {
            let next_val = cur_val.map(|v| v.square());
            let next_var = if i == self.num_squarings - 1 {
                // Last squaring: allocate as public input
                cs.alloc_input(
                    || format!("sq_{i}"),
                    || next_val.ok_or(bellman::SynthesisError::AssignmentMissing),
                )?
            } else {
                cs.alloc(
                    || format!("sq_{i}"),
                    || next_val.ok_or(bellman::SynthesisError::AssignmentMissing),
                )?
            };
            cs.enforce(
                || format!("sq_constraint_{i}"),
                |lc| lc + cur_var,
                |lc| lc + cur_var,
                |lc| lc + next_var,
            );
            cur_val = next_val;
            cur_var = next_var;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Shared setup: creates GPU context + trusted setup params (cached per bench)
// ---------------------------------------------------------------------------
struct BenchSetup {
    params: bellman::groth16::Parameters<Bls12>,
    ppk: PreparedProvingKey<Bls12>,
    gpu: GpuContext<Bls12>,
    num_squarings: usize,
}

fn setup(num_squarings: usize) -> BenchSetup {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut rng = OsRng;

    let setup_circuit = RepeatedSquaringCircuit::<Scalar> {
        x: None,
        num_squarings,
    };
    let params =
        bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
            .expect("param gen failed");

    let ppk = prover::prepare_proving_key::<Bls12, Bls12>(&params);
    let gpu = rt.block_on(GpuContext::<Bls12>::new()).expect("gpu init failed");

    BenchSetup {
        params,
        ppk,
        gpu,
        num_squarings,
    }
}

// ---------------------------------------------------------------------------
// Benchmark: full end-to-end proof generation at various circuit sizes
// ---------------------------------------------------------------------------
fn bench_full_proof(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let x = Scalar::from(3u64);

    let mut group = c.benchmark_group("proof");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for &num_sq in &[2, 1_000, 10_000, 100_000] {
        let bs = setup(num_sq);
        group.bench_function(format!("n={num_sq}"), |b| {
            b.iter(|| {
                let circuit = RepeatedSquaringCircuit::<Scalar> {
                    x: Some(x),
                    num_squarings: bs.num_squarings,
                };
                let mut rng = OsRng;
                rt.block_on(prover::create_proof::<Bls12, Bls12, _, _>(
                    circuit,
                    &bs.params,
                    &bs.ppk,
                    &bs.gpu,
                    &mut rng,
                ))
                .expect("proof failed");
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: H polynomial computation (NTT pipeline)
// ---------------------------------------------------------------------------
fn bench_h_poly(c: &mut Criterion) {
    let bs = setup(2);
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("ntt");
    group.sample_size(10);

    for &n in &[8, 1024, 16384] {
        let mut rng = OsRng;
        let a: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();
        let b: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();
        let cv: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();

        group.bench_function(format!("h_poly_n={n}"), |bench| {
            bench.iter(|| {
                rt.block_on(prover::compute_h_poly::<Bls12>(&bs.gpu, &a, &b, &cv))
                    .expect("h_poly failed");
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: single MSM G1 at various sizes
// ---------------------------------------------------------------------------
fn bench_msm_g1(c: &mut Criterion) {
    let bs = setup(2);
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("msm_g1");
    group.sample_size(10);

    for &size in &[100, 1_000, 10_000, 100_000] {
        let mut rng = OsRng;
        let bases: Vec<<Bls12 as GpuCurve>::G1Affine> = (0..size)
            .map(|_| <Bls12 as GpuCurve>::G1::random(&mut rng).to_affine())
            .collect();
        let scalars: Vec<Scalar> = (0..size).map(|_| Scalar::random(&mut rng)).collect();

        group.bench_function(format!("n={size}"), |b| {
            b.iter(|| {
                rt.block_on(prover::gpu_msm_g1::<Bls12>(&bs.gpu, &bases, &scalars))
                    .expect("msm failed");
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: MSM batch (5 MSMs as in actual proof) at various sizes
// ---------------------------------------------------------------------------
fn bench_msm_batch(c: &mut Criterion) {
    let bs = setup(2);
    let rt = tokio::runtime::Runtime::new().unwrap();

    fn make_g1_bases(rng: &mut OsRng, count: usize) -> Vec<<Bls12 as GpuCurve>::G1Affine> {
        (0..count)
            .map(|_| <Bls12 as GpuCurve>::G1::random(&mut *rng).to_affine())
            .collect()
    }
    fn make_g2_bases(rng: &mut OsRng, count: usize) -> Vec<<Bls12 as GpuCurve>::G2Affine> {
        (0..count)
            .map(|_| <Bls12 as GpuCurve>::G2::random(&mut *rng).to_affine())
            .collect()
    }
    fn make_scalars(rng: &mut OsRng, count: usize) -> Vec<Scalar> {
        (0..count).map(|_| Scalar::random(&mut *rng)).collect()
    }

    let mut group = c.benchmark_group("msm_batch");
    group.sample_size(10);

    for &n in &[100, 1_000, 10_000, 100_000] {
        let mut rng = OsRng;
        let a_bases = make_g1_bases(&mut rng, n);
        let a_scalars = make_scalars(&mut rng, n);
        let b1_bases = make_g1_bases(&mut rng, n);
        let b_scalars = make_scalars(&mut rng, n);
        let l_bases = make_g1_bases(&mut rng, n);
        let l_scalars = make_scalars(&mut rng, n);
        let h_bases = make_g1_bases(&mut rng, n);
        let h_scalars = make_scalars(&mut rng, n);
        let b2_bases = make_g2_bases(&mut rng, n);
        let b2_scalars = make_scalars(&mut rng, n);

        group.bench_function(format!("5x{n}"), |b| {
            b.iter(|| {
                rt.block_on(prover::gpu_msm_batch::<Bls12>(
                    &bs.gpu,
                    &a_bases,
                    &a_scalars,
                    &b1_bases,
                    &b_scalars,
                    &l_bases,
                    &l_scalars,
                    &h_bases,
                    &h_scalars,
                    &b2_bases,
                    &b2_scalars,
                ))
                .expect("msm batch failed");
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: CPU bucket sorting
// ---------------------------------------------------------------------------
fn bench_bucket_sorting(c: &mut Criterion) {
    let mut group = c.benchmark_group("bucket_sorting");

    for &size in &[1_000, 10_000, 100_000] {
        let mut rng = OsRng;
        let scalars: Vec<Scalar> = (0..size).map(|_| Scalar::random(&mut rng)).collect();

        group.bench_function(format!("n={size}"), |b| {
            b.iter(|| {
                compute_bucket_sorting::<Bls12>(&scalars);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// MASP Sapling Output circuit helpers
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Benchmark: MASP Sapling Output proof (real-world circuit, ~31K constraints)
// ---------------------------------------------------------------------------
fn bench_sapling_output(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut rng = OsRng;

    let setup_circuit = SaplingOutputCircuit {
        value_commitment: None,
        asset_identifier: vec![None; 256],
        payment_address: None,
        commitment_randomness: None,
        esk: None,
    };
    let params =
        bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
            .expect("param gen failed");
    let ppk = prover::prepare_proving_key::<Bls12, Bls12>(&params);
    let gpu = rt.block_on(GpuContext::<Bls12>::new()).expect("gpu init failed");

    let mut group = c.benchmark_group("sapling_output");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    group.bench_function("proof", |b| {
        b.iter(|| {
            let circuit = sample_sapling_output_circuit();
            let mut rng = OsRng;
            rt.block_on(prover::create_proof::<Bls12, Bls12, _, _>(
                circuit, &params, &ppk, &gpu, &mut rng,
            ))
            .expect("proof failed");
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_full_proof,
    bench_h_poly,
    bench_msm_g1,
    bench_msm_batch,
    bench_bucket_sorting,
    bench_sapling_output,
);
criterion_main!(benches);
