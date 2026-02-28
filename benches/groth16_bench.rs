use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use blstrs::{Bls12, Scalar};
use ff::Field;
use group::{Curve, Group};
use rand_core::OsRng;

use webgpu_groth16::bellman;
use webgpu_groth16::bucket::compute_bucket_sorting;
use webgpu_groth16::gpu::GpuContext;
use webgpu_groth16::gpu::curve::GpuCurve;
use webgpu_groth16::prover;
use webgpu_groth16::prover::PreparedProvingKey;

// ---------------------------------------------------------------------------
// Dummy circuit: proves knowledge of x such that x^3 = y
// ---------------------------------------------------------------------------
struct DummyCircuit<S: ff::PrimeField> {
    x: Option<S>,
    y: Option<S>,
}

impl<S: ff::PrimeField> bellman::Circuit<S> for DummyCircuit<S> {
    fn synthesize<CS: bellman::ConstraintSystem<S>>(
        self,
        cs: &mut CS,
    ) -> Result<(), bellman::SynthesisError> {
        let y_val = self.y;
        let y = cs.alloc_input(
            || "y",
            || y_val.ok_or(bellman::SynthesisError::AssignmentMissing),
        )?;

        let x_val = self.x;
        let x = cs.alloc(
            || "x",
            || x_val.ok_or(bellman::SynthesisError::AssignmentMissing),
        )?;

        let x_sq_val = x_val.map(|v| v.square());
        let x_sq = cs.alloc(
            || "x_sq",
            || x_sq_val.ok_or(bellman::SynthesisError::AssignmentMissing),
        )?;
        cs.enforce(|| "x * x = x_sq", |lc| lc + x, |lc| lc + x, |lc| lc + x_sq);
        cs.enforce(|| "x_sq * x = y", |lc| lc + x_sq, |lc| lc + x, |lc| lc + y);

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
}

fn setup() -> BenchSetup {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut rng = OsRng;

    let setup_circuit = DummyCircuit::<Scalar> { x: None, y: None };
    let params =
        bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
            .expect("param gen failed");

    let ppk = prover::prepare_proving_key::<Bls12, Bls12>(&params);
    let gpu = rt.block_on(GpuContext::<Bls12>::new()).expect("gpu init failed");

    BenchSetup { params, ppk, gpu }
}

// ---------------------------------------------------------------------------
// Benchmark: full end-to-end proof generation
// ---------------------------------------------------------------------------
fn bench_full_proof(c: &mut Criterion) {
    let bs = setup();
    let rt = tokio::runtime::Runtime::new().unwrap();

    let x = Scalar::from(3u64);
    let y = x.square() * x; // x^3

    let mut group = c.benchmark_group("proof");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    group.bench_function("full_proof", |b| {
        b.iter(|| {
            let circuit = DummyCircuit::<Scalar> {
                x: Some(x),
                y: Some(y),
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
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: H polynomial computation (NTT pipeline)
// ---------------------------------------------------------------------------
fn bench_h_poly(c: &mut Criterion) {
    let bs = setup();
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Create some random scalar vectors of size 8 (next_power_of_two of the dummy circuit)
    let n = 8;
    let mut rng = OsRng;
    let a: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();
    let b: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();
    let cv: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();

    let mut group = c.benchmark_group("ntt");
    group.sample_size(10);
    group.bench_function("h_poly", |bench| {
        bench.iter(|| {
            rt.block_on(prover::compute_h_poly::<Bls12>(&bs.gpu, &a, &b, &cv))
                .expect("h_poly failed");
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: single MSM G1 at n=100
// ---------------------------------------------------------------------------
fn bench_msm_g1(c: &mut Criterion) {
    let bs = setup();
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("msm_g1");
    group.sample_size(10);

    let size = 100;
    let mut rng = OsRng;
    let bases: Vec<<Bls12 as GpuCurve>::G1Affine> = (0..size)
        .map(|_| <Bls12 as GpuCurve>::G1::random(&mut rng).to_affine())
        .collect();
    let scalars: Vec<Scalar> = (0..size).map(|_| Scalar::random(&mut rng)).collect();

    group.bench_function("n=100", |b| {
        b.iter(|| {
            rt.block_on(prover::gpu_msm_g1::<Bls12>(&bs.gpu, &bases, &scalars))
                .expect("msm failed");
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: MSM batch (5 MSMs as in actual proof)
// ---------------------------------------------------------------------------
fn bench_msm_batch(c: &mut Criterion) {
    let bs = setup();
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut rng = OsRng;
    let n = 100;

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

    let mut group = c.benchmark_group("msm_batch");
    group.sample_size(10);
    group.bench_function("5x100", |b| {
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
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: CPU bucket sorting
// ---------------------------------------------------------------------------
fn bench_bucket_sorting(c: &mut Criterion) {
    let mut group = c.benchmark_group("bucket_sorting");

    for &size in &[1_000, 10_000] {
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

criterion_group!(
    benches,
    bench_full_proof,
    bench_h_poly,
    bench_msm_g1,
    bench_msm_batch,
    bench_bucket_sorting,
);
criterion_main!(benches);
