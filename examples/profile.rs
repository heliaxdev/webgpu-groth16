//! GPU profiling harness using wgpu-profiler + tracy.
//!
//! Usage:
//!   cargo run --release --example profile --features profiling -- [NUM_SQUARINGS] [ITERATIONS]
//!
//! Then connect Tracy profiler to see the flamegraph.
//!
//! Defaults: NUM_SQUARINGS=10000, ITERATIONS=5

use std::io::Read as _;
use std::time::Instant;

use blstrs::{Bls12, Scalar};
use rand_core::OsRng;

use webgpu_groth16::bellman;
use webgpu_groth16::gpu::GpuContext;
use webgpu_groth16::prover;

// ---------------------------------------------------------------------------
// Repeated-squaring circuit (same as benches/groth16_bench.rs)
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_squarings: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    eprintln!("=== Groth16 GPU Profiling ===");
    eprintln!("  Connect Tracy profiler to view the flamegraph");
    eprintln!("  constraints: {num_squarings}");
    eprintln!("  iterations:  {iterations}");

    // --- Trusted setup (one-time cost) ---
    let t_setup = Instant::now();
    let mut rng = OsRng;
    let setup_circuit = RepeatedSquaringCircuit::<Scalar> {
        x: None,
        num_squarings,
    };
    let params =
        bellman::groth16::generate_random_parameters::<Bls12, _, _>(setup_circuit, &mut rng)
            .expect("param gen failed");
    let ppk = prover::prepare_proving_key::<Bls12, Bls12>(&params);
    eprintln!("  setup:       {:?}", t_setup.elapsed());

    // --- GPU init ---
    let t_gpu = Instant::now();
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = rt
        .block_on(GpuContext::<Bls12>::new())
        .expect("gpu init failed");
    eprintln!("  gpu init:    {:?}", t_gpu.elapsed());

    // --- Warmup (shader compilation) ---
    eprintln!("  warmup...");
    {
        tracy_client::Client::start();
        let circuit = RepeatedSquaringCircuit::<Scalar> {
            x: Some(Scalar::from(3u64)),
            num_squarings,
        };
        rt.block_on(prover::create_proof::<Bls12, Bls12, _, _>(
            circuit, &params, &ppk, &gpu, &mut rng,
        ))
        .expect("warmup proof failed");
        gpu.end_profiler_frame();
        // Drain warmup results.
        let _ = gpu.process_profiler_results();
    }

    // --- Profiled iterations ---
    eprintln!("  profiling {iterations} iterations...");
    let x = Scalar::from(3u64);
    let t_total = Instant::now();
    for i in 0..iterations {
        tracy_client::Client::running().expect("tracy client").frame_mark();

        let t_iter = Instant::now();
        let circuit = RepeatedSquaringCircuit::<Scalar> {
            x: Some(x),
            num_squarings,
        };
        let _proof = rt
            .block_on(prover::create_proof::<Bls12, Bls12, _, _>(
                circuit, &params, &ppk, &gpu, &mut rng,
            ))
            .expect("proof failed");

        gpu.end_profiler_frame();

        if let Some(results) = gpu.process_profiler_results() {
            eprintln!("  iter {}: {:?} (GPU breakdown below)", i + 1, t_iter.elapsed());
            print_gpu_results(&results, 2);
        } else {
            eprintln!("  iter {}: {:?} (GPU results pending)", i + 1, t_iter.elapsed());
        }
    }
    eprintln!("  total:   {:?} ({iterations} proofs)", t_total.elapsed());
    eprintln!(
        "  avg:     {:?}/proof",
        t_total.elapsed() / iterations as u32
    );

    eprintln!();
    eprintln!("Press Enter to exit (keep Tracy open to inspect results)...");
    let _ = std::io::stdin().read(&mut [0u8]);
}

fn print_gpu_results(results: &[wgpu_profiler::GpuTimerQueryResult], indent: usize) {
    let pad = " ".repeat(indent);
    for r in results {
        if let Some(ref time) = r.time {
            let duration_s = time.end - time.start;
            if duration_s < 0.0 {
                eprintln!("{pad}{}: <invalid: negative duration>", r.label);
            } else if duration_s < 0.001 {
                eprintln!("{pad}{}: {:.1} us", r.label, duration_s * 1_000_000.0);
            } else {
                eprintln!("{pad}{}: {:.2} ms", r.label, duration_s * 1_000.0);
            }
        } else {
            eprintln!("{pad}{}: (no timing data)", r.label);
        }
        if !r.nested_queries.is_empty() {
            print_gpu_results(&r.nested_queries, indent + 2);
        }
    }
}
