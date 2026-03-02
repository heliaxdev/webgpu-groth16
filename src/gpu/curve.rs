//! Curve abstraction for GPU proving.
//!
//! This module defines [`GpuCurve`], the curve-agnostic interface used by the
//! prover, MSM bucket builder, and GPU dispatch code.
//!
//! Curve-specific serialization/layout details, shader source wiring, and
//! arithmetic bridges are implemented in submodules (for example
//! `src/gpu/curve/bls12_381.rs`).

use ff::{PrimeField, PrimeFieldBits};
use group::Group;

/// Per-scalar decomposition strategy for G1 MSM bucket construction.
pub enum G1MsmDecomposition {
    Standard {
        windows: Vec<(u32, bool)>,
    },
    Glv {
        k1_windows: Vec<(u32, bool)>,
        k1_neg: bool,
        k2_windows: Vec<(u32, bool)>,
        k2_neg: bool,
    },
}

pub type GlvWindowDecomposition = (Vec<(u32, bool)>, bool, Vec<(u32, bool)>, bool);

/// Abstraction over a pairing-friendly curve for GPU-accelerated proving.
pub trait GpuCurve: 'static {
    type Engine: pairing::Engine;

    type Scalar: PrimeField + PrimeFieldBits;
    type G1: Group<Scalar = Self::Scalar>;
    type G2: Group<Scalar = Self::Scalar>;
    type G1Affine;
    type G2Affine;

    /// Size of one Fq element in GPU buffer layout.
    const FQ_GPU_BYTES: usize;
    /// Size of one padded Fq element for WGSL `@size(128)` structs.
    const FQ_GPU_PADDED_BYTES: usize;
    /// Size of one G1 point in GPU buffer layout.
    const G1_GPU_BYTES: usize;
    /// Size of one G2 point in GPU buffer layout.
    const G2_GPU_BYTES: usize;

    /// Workgroup size for scalar-domain kernels (NTT global, montgomery, shifts).
    const SCALAR_WORKGROUP_SIZE: u32;
    /// Elements processed per tile-local NTT workgroup.
    const NTT_TILE_SIZE: u32;
    /// Workgroup size for MSM kernels.
    const MSM_WORKGROUP_SIZE: u32;
    /// Phase-1 chunks per window for G1 MSM subsum.
    const G1_SUBSUM_CHUNKS_PER_WINDOW: u32;
    /// Phase-1 chunks per window for G2 MSM subsum.
    const G2_SUBSUM_CHUNKS_PER_WINDOW: u32;
    /// Maximum bucket chunk size used to split large buckets for GPU load balancing.
    const MSM_MAX_CHUNK_SIZE: u32;
    /// Bit mask used to encode point-sign metadata in packed MSM base indices.
    const MSM_INDEX_SIGN_BIT: u32;

    const NTT_SOURCE: &'static str;
    const NTT_FUSED_SOURCE: &'static str;
    const MSM_G1_AGG_SOURCE: &'static str;
    const MSM_G1_SUBSUM_SOURCE: &'static str;
    const MSM_G2_AGG_SOURCE: &'static str;
    const MSM_G2_SUBSUM_SOURCE: &'static str;
    #[cfg(test)]
    const TEST_SHADER_G1_SOURCE: &'static str;
    #[cfg(test)]
    const TEST_SHADER_G2_SOURCE: &'static str;
    const POLY_OPS_SOURCE: &'static str;

    /// Whether this curve supports GLV acceleration for G1 MSM.
    const HAS_G1_GLV: bool;

    fn serialize_g1(point: &Self::G1Affine) -> Vec<u8>;
    fn serialize_g2(point: &Self::G2Affine) -> Vec<u8>;
    fn serialize_scalar(s: &Self::Scalar) -> Vec<u8>;
    fn deserialize_scalar(bytes: &[u8]) -> anyhow::Result<Self::Scalar>;
    fn deserialize_g1(bytes: &[u8]) -> anyhow::Result<Self::G1>;
    fn deserialize_g2(bytes: &[u8]) -> anyhow::Result<Self::G2>;

    fn scalar_to_windows(s: &Self::Scalar, c: usize) -> Vec<u32>;

    fn scalar_to_signed_windows(s: &Self::Scalar, c: usize) -> Vec<(u32, bool)> {
        let unsigned = Self::scalar_to_windows(s, c);
        let half = 1u64 << (c - 1);
        let full = 1u64 << c;
        let mut result = Vec::with_capacity(unsigned.len() + 1);
        let mut carry: u64 = 0;

        for &w in &unsigned {
            let val = w as u64 + carry;
            carry = 0;
            if val >= half {
                let abs = full - val;
                result.push((abs as u32, true));
                carry = 1;
            } else {
                result.push((val as u32, false));
            }
        }

        if carry > 0 {
            result.push((1, false));
        }

        result
    }

    fn decompose_g1_msm_scalar(s: &Self::Scalar, c: usize) -> G1MsmDecomposition {
        G1MsmDecomposition::Standard {
            windows: Self::scalar_to_signed_windows(s, c),
        }
    }

    /// Optional fast-path decomposition for GLV-capable curves.
    ///
    /// Returns `(k1_windows, k1_neg, k2_windows, k2_neg)` when available.
    fn decompose_g1_msm_scalar_glv_windows(
        s: &Self::Scalar,
        c: usize,
    ) -> Option<GlvWindowDecomposition> {
        match Self::decompose_g1_msm_scalar(s, c) {
            G1MsmDecomposition::Glv {
                k1_windows,
                k1_neg,
                k2_windows,
                k2_neg,
            } => Some((k1_windows, k1_neg, k2_windows, k2_neg)),
            G1MsmDecomposition::Standard { .. } => None,
        }
    }

    fn g1_msm_bucket_width(n: usize) -> usize {
        let _ = n;
        Self::bucket_width()
    }

    fn g1_endomorphism_base_bytes(_base_bytes: &[u8]) -> Option<Vec<u8>> {
        None
    }

    fn negate_g1_base_bytes(point_bytes: &mut [u8]) {
        let p = Self::deserialize_g1(point_bytes)
            .expect("deserialize_g1 failed in negate_g1_base_bytes");
        let neg = Self::sub_g1_proj(&Self::g1_identity(), &p);
        let neg_aff = Self::proj_to_affine_g1(&neg);
        let bytes = Self::serialize_g1(&neg_aff);
        point_bytes.copy_from_slice(&bytes);
    }

    fn bucket_width() -> usize;

    fn g2_bucket_width() -> usize {
        8
    }

    fn glv_bucket_width() -> usize {
        13
    }

    fn root_of_unity(n: usize) -> Self::Scalar;

    fn g1_identity() -> Self::G1 {
        Self::G1::identity()
    }
    fn g2_identity() -> Self::G2 {
        Self::G2::identity()
    }

    fn affine_to_proj_g1(p: &Self::G1Affine) -> Self::G1;
    fn affine_to_proj_g2(p: &Self::G2Affine) -> Self::G2;
    fn proj_to_affine_g1(p: &Self::G1) -> Self::G1Affine;
    fn proj_to_affine_g2(p: &Self::G2) -> Self::G2Affine;

    fn add_g1_proj(a: &Self::G1, b: &Self::G1) -> Self::G1;
    fn sub_g1_proj(a: &Self::G1, b: &Self::G1) -> Self::G1;
    fn add_g2_proj(a: &Self::G2, b: &Self::G2) -> Self::G2;

    fn mul_g1_scalar(a: &Self::G1Affine, b: &Self::Scalar) -> Self::G1;
    fn mul_g2_scalar(a: &Self::G2Affine, b: &Self::Scalar) -> Self::G2;
    fn mul_g1_proj_scalar(a: &Self::G1, b: &Self::Scalar) -> Self::G1;
}

pub mod bls12_381;

#[cfg(test)]
#[path = "curve_tests.rs"]
mod tests;
