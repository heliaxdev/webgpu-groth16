//! Pre-serialized proving key for GPU dispatch.
//!
//! Converts proving key bases to GPU-friendly byte representations once,
//! amortizing serialization cost across multiple proofs. Includes GLV
//! endomorphism bases φ(P) for G1 sets.

use crate::bellman;
use crate::glv;
use crate::gpu::curve::{G1_GPU_BYTES, G2_GPU_BYTES, GpuCurve};

/// Pre-serialized proving key bases for GPU. Avoids re-serialization per proof.
///
/// Includes GLV endomorphism bases φ(P) for G1 sets, pre-computed once to
/// amortize the endomorphism cost across proofs.
pub struct PreparedProvingKey<G: GpuCurve> {
    pub a_bytes: Vec<u8>,
    pub a_phi_bytes: Vec<u8>,
    pub b_g1_bytes: Vec<u8>,
    pub b_g1_phi_bytes: Vec<u8>,
    pub l_bytes: Vec<u8>,
    pub l_phi_bytes: Vec<u8>,
    pub h_bytes: Vec<u8>,
    pub h_phi_bytes: Vec<u8>,
    pub b_g2_bytes: Vec<u8>,
    _marker: std::marker::PhantomData<G>,
}

pub(crate) fn serialize_g1_bases<G: GpuCurve>(bases: &[G::G1Affine]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(bases.len() * G1_GPU_BYTES);
    for base in bases {
        bytes.extend_from_slice(&G::serialize_g1(base));
    }
    bytes
}

pub(crate) fn serialize_g1_phi_bases<G: GpuCurve>(bases: &[G::G1Affine]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(bases.len() * G1_GPU_BYTES);
    for base in bases {
        let base_bytes = G::serialize_g1(base);
        bytes.extend_from_slice(&glv::endomorphism_g1_bytes(&base_bytes));
    }
    bytes
}

pub(crate) fn serialize_g2_bases<G: GpuCurve>(bases: &[G::G2Affine]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(bases.len() * G2_GPU_BYTES);
    for base in bases {
        bytes.extend_from_slice(&G::serialize_g2(base));
    }
    bytes
}

/// Interleave base bytes and phi bytes into [P₀, φ(P₀), P₁, φ(P₁), ...] layout.
pub(crate) fn interleave_glv_bases(
    bases_bytes: &[u8],
    phi_bytes: &[u8],
    point_size: usize,
) -> Vec<u8> {
    let n = bases_bytes.len() / point_size;
    debug_assert_eq!(bases_bytes.len(), n * point_size);
    debug_assert_eq!(phi_bytes.len(), n * point_size);
    let mut combined = Vec::with_capacity(n * 2 * point_size);
    for i in 0..n {
        let start = i * point_size;
        combined.extend_from_slice(&bases_bytes[start..start + point_size]);
        combined.extend_from_slice(&phi_bytes[start..start + point_size]);
    }
    combined
}

pub fn prepare_proving_key<E, G>(pk: &bellman::groth16::Parameters<E>) -> PreparedProvingKey<G>
where
    E: pairing::MultiMillerLoop,
    G: GpuCurve<
            Engine = E,
            Scalar = E::Fr,
            G1 = E::G1,
            G2 = E::G2,
            G1Affine = E::G1Affine,
            G2Affine = E::G2Affine,
        >,
{
    PreparedProvingKey {
        a_bytes: serialize_g1_bases::<G>(&pk.a),
        a_phi_bytes: serialize_g1_phi_bases::<G>(&pk.a),
        b_g1_bytes: serialize_g1_bases::<G>(&pk.b_g1),
        b_g1_phi_bytes: serialize_g1_phi_bases::<G>(&pk.b_g1),
        l_bytes: serialize_g1_bases::<G>(&pk.l),
        l_phi_bytes: serialize_g1_phi_bases::<G>(&pk.l),
        h_bytes: serialize_g1_bases::<G>(&pk.h),
        h_phi_bytes: serialize_g1_phi_bases::<G>(&pk.h),
        b_g2_bytes: serialize_g2_bases::<G>(&pk.b_g2),
        _marker: std::marker::PhantomData,
    }
}
