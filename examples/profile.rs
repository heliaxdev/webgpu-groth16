//! GPU profiling / benchmarking harness.
//!
//! Without features (wall-clock timing only):
//!   cargo run --release --example profile -- [NUM_SQUARINGS] [ITERATIONS]
//!
//! With wgpu-profiler (GPU timing + chrome trace output):
//!   cargo run --release --example profile --features profiling -- [NUM_SQUARINGS] [ITERATIONS]
//!   Then open the generated `profile.json` in chrome://tracing or https://ui.perfetto.dev
//!
//! Defaults: NUM_SQUARINGS=10000, ITERATIONS=5

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
    #[cfg(feature = "profiling")]
    eprintln!("  GPU profiling enabled — trace will be written to profile.json");
    #[cfg(not(feature = "profiling"))]
    eprintln!("  Wall-clock timing only (enable 'profiling' feature for GPU breakdown)");
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
        let circuit = RepeatedSquaringCircuit::<Scalar> {
            x: Some(Scalar::from(3u64)),
            num_squarings,
        };
        rt.block_on(prover::create_proof::<Bls12, Bls12, _, _>(
            circuit, &params, &ppk, &gpu, &mut rng,
        ))
        .expect("warmup proof failed");

        #[cfg(feature = "profiling")]
        {
            gpu.end_profiler_frame();
            let _ = gpu.process_profiler_results();
        }
    }

    // --- Profiled iterations ---
    eprintln!("  profiling {iterations} iterations...");
    let x = Scalar::from(3u64);
    let t_total = Instant::now();

    #[cfg(feature = "profiling")]
    let mut all_profiling_data = Vec::new();

    for i in 0..iterations {
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

        #[cfg(feature = "profiling")]
        {
            gpu.end_profiler_frame();
            if let Some(results) = gpu.process_profiler_results() {
                eprintln!(
                    "  iter {}: {:?} (GPU breakdown below)",
                    i + 1,
                    t_iter.elapsed()
                );
                print_gpu_results(&results, 2);
                all_profiling_data.extend(results);
            } else {
                eprintln!(
                    "  iter {}: {:?} (GPU results pending)",
                    i + 1,
                    t_iter.elapsed()
                );
            }
        }

        #[cfg(not(feature = "profiling"))]
        eprintln!("  iter {}: {:?}", i + 1, t_iter.elapsed());
    }
    eprintln!("  total:   {:?} ({iterations} proofs)", t_total.elapsed());
    eprintln!(
        "  avg:     {:?}/proof",
        t_total.elapsed() / iterations as u32
    );

    #[cfg(feature = "profiling")]
    {
        let trace_path = std::path::Path::new("profile.json");
        write_chrometrace(trace_path, &all_profiling_data)
            .expect("failed to write chrome trace");
        eprintln!();
        eprintln!("  Trace written to {}", trace_path.display());
        eprintln!("  Open in chrome://tracing or https://ui.perfetto.dev");
    }
}

/// Compute the time range for a result.
///
/// For parent scopes (with children), always synthesize from children's
/// min(start)..max(end) — encoder-level timestamps from `scope()` are
/// unreliable on Metal (negative/zero/bogus durations).
///
/// For leaf scopes (no children), use the direct timestamp if it has a
/// positive duration; otherwise return None to skip broken events.
#[cfg(feature = "profiling")]
fn effective_time(r: &wgpu_profiler::GpuTimerQueryResult) -> Option<std::ops::Range<f64>> {
    if !r.nested_queries.is_empty() {
        // Parent scope: always synthesize from children.
        let mut start = f64::MAX;
        let mut end = f64::MIN;
        for child in &r.nested_queries {
            if let Some(ct) = effective_time(child) {
                start = start.min(ct.start);
                end = end.max(ct.end);
            }
        }
        if start < end { Some(start..end) } else { None }
    } else if let Some(ref t) = r.time {
        // Leaf scope: use direct timestamp only if valid.
        let dur = t.end - t.start;
        if dur > 0.0 && t.start > 0.0 {
            Some(t.clone())
        } else {
            None
        }
    } else {
        None
    }
}

#[cfg(feature = "profiling")]
fn print_gpu_results(results: &[wgpu_profiler::GpuTimerQueryResult], indent: usize) {
    let pad = " ".repeat(indent);
    for r in results {
        if let Some(time) = effective_time(r) {
            let duration_s = time.end - time.start;
            if duration_s < 0.001 {
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

/// Find the earliest timestamp across all results (recursively).
#[cfg(feature = "profiling")]
fn find_min_time(results: &[wgpu_profiler::GpuTimerQueryResult]) -> f64 {
    let mut min_t = f64::MAX;
    for r in results {
        if let Some(t) = effective_time(r) {
            min_t = min_t.min(t.start);
        }
        min_t = min_t.min(find_min_time(&r.nested_queries));
    }
    min_t
}

/// Write a Chrome trace JSON file, synthesizing parent spans from children
/// when the parent has no direct timestamps.  All timestamps are normalised
/// relative to the earliest event so Perfetto shows human-readable times.
#[cfg(feature = "profiling")]
fn write_chrometrace(
    path: &std::path::Path,
    results: &[wgpu_profiler::GpuTimerQueryResult],
) -> std::io::Result<()> {
    use std::io::Write;
    let t0 = find_min_time(results);
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(file, "{{\n\"traceEvents\": [\n")?;
    let mut first = true;
    for r in results {
        write_trace_event(&mut file, r, &mut first, t0)?;
    }
    write!(file, "\n]\n}}\n")?;
    Ok(())
}

#[cfg(feature = "profiling")]
fn tid_int(r: &wgpu_profiler::GpuTimerQueryResult) -> u64 {
    let raw = format!("{:?}", r.tid);
    raw.chars()
        .filter(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse::<u64>()
        .unwrap_or(1)
}

#[cfg(feature = "profiling")]
fn write_trace_event(
    w: &mut impl std::io::Write,
    r: &wgpu_profiler::GpuTimerQueryResult,
    first: &mut bool,
    t0: f64,
) -> std::io::Result<()> {
    let has_children = !r.nested_queries.is_empty();
    if let Some(time) = effective_time(r) {
        let ts_us = (time.start - t0) * 1_000_000.0;
        let pid = r.pid;
        let tid = tid_int(r);

        if has_children {
            // Use B/E (begin/end) for parent scopes to avoid overlapping
            // complete events on the same track.  Nudge the B timestamp
            // slightly before and E slightly after children so Perfetto
            // doesn't treat them as misplaced.
            let b_us = ts_us - 0.01;
            let comma = if *first { "" } else { ",\n" };
            write!(
                w,
                "{comma}{{ \"pid\":{pid}, \"tid\":{tid}, \"ts\":{b_us}, \"ph\":\"B\", \"name\":\"{}\" }}",
                r.label,
            )?;
            *first = false;

            for child in &r.nested_queries {
                write_trace_event(w, child, first, t0)?;
            }

            let e_us = (time.end - t0) * 1_000_000.0 + 0.01;
            write!(
                w,
                ",\n{{ \"pid\":{pid}, \"tid\":{tid}, \"ts\":{e_us}, \"ph\":\"E\", \"name\":\"{}\" }}",
                r.label,
            )?;
        } else {
            // Leaf event: use X (complete) — no overlap risk.
            let dur_us = (time.end - time.start) * 1_000_000.0;
            let comma = if *first { "" } else { ",\n" };
            write!(
                w,
                "{comma}{{ \"pid\":{pid}, \"tid\":{tid}, \"ts\":{ts_us}, \"dur\":{dur_us}, \"ph\":\"X\", \"name\":\"{}\" }}",
                r.label,
            )?;
            *first = false;
        }
    } else {
        // No timing at all — still recurse into children
        for child in &r.nested_queries {
            write_trace_event(w, child, first, t0)?;
        }
    }
    Ok(())
}
