# wgpu prover

right now, this is woefully unoptimized. on a macbook m3 max, it takes 20 seconds to
prove the dummy circuit in `test_gpu_groth16_prover`. the cpu equivalent is instant.
moreoever, in `src/prover.rs`, we make assumptions about the underlying field. we
cannot do this. the implementation must use the `GpuCurve` trait in `src/gpu/curve.rs`
for generically operating over finite fields/elliptic curves.
