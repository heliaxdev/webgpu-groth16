//! GPU-accelerated Groth16 zero-knowledge proof system for BLS12-381.
//!
//! The two most expensive proof operations — **MSM** (multi-scalar multiplication)
//! and **NTT** (number theoretic transform) — are offloaded to GPU compute shaders
//! written in WGSL and dispatched via [wgpu](https://docs.rs/wgpu).
//!
//! # Architecture
//!
//! - [`gpu`] — WebGPU context, pipeline management, and kernel dispatchers
//!   (MSM, NTT, H-polynomial). Field arithmetic uses 13-bit limbs for Fq
//!   and 32-bit limbs for Fr.
//! - [`prover`] — Groth16 proof construction orchestration: circuit synthesis,
//!   witness evaluation, H-polynomial computation, and MSM scheduling.
//! - [`bucket`] — Pippenger bucket sorting with signed-digit scalar decomposition
//!   and sub-bucket chunking for GPU load balancing.
//! - [`glv`] — GLV endomorphism for BLS12-381 G1, decomposing 255-bit scalars
//!   into two ~128-bit sub-scalars to halve MSM windows.
//!
//! # MSM Pipeline
//!
//! G1 MSMs use GLV-accelerated Pippenger: each scalar is decomposed via the
//! BLS12-381 endomorphism into `k1·P + k2·φ(P)`, halving the number of windows.
//! The GPU pipeline runs 5 kernels per MSM:
//!
//! `to_montgomery → aggregate_buckets → [reduce_sub_buckets] → weight_buckets → subsum`
//!
//! Persistent GPU bases ([`prover::GpuProvingKey`]) can be pre-uploaded once and
//! reused across proofs, eliminating per-proof base transfers.

pub mod bucket;
pub mod glv;
pub mod gpu;
pub mod prover;

#[cfg(feature = "bellman-provider-bellman")]
pub use bellman;

#[cfg(feature = "bellman-provider-nam-bellperson")]
pub use nam_bellperson as bellman;
