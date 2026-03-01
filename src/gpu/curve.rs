//! CPU ↔ GPU serialization for BLS12-381 curve elements.
//!
//! The GPU shaders operate on a **30×13-bit limb** representation of Fq elements
//! (see [`fq_bytes_to_13bit`]), while the CPU uses the standard 48-byte (384-bit)
//! big-endian format from the Zcash / blstrs libraries. This module bridges the two
//! by providing:
//!
//! - **[`GpuCurve`]** — a trait abstracting serialization, scalar decomposition,
//!   NTT support, and point arithmetic so the prover is not hard-wired to one curve
//!   implementation.
//! - **Serialize / deserialize** for G1, G2, and scalar field elements, handling the
//!   endianness flip (Zcash BE → GPU LE), the 13-bit limb packing, and the
//!   Jacobian Z-coordinate insertion (affine → projective with Z = 1).
//! - **Scalar window decomposition** (unsigned and signed-digit) used by the
//!   Pippenger MSM bucket sort.
//!
//! # Byte-layout cheat sheet
//!
//! | Element | CPU (blstrs)              | GPU (WGSL)                           |
//! |---------|---------------------------|--------------------------------------|
//! | Fq      | 48 B, big-endian          | 120 B, 30 × u32 LE (13-bit limbs)   |
//! | G1      | 96 B uncompressed (x‖y)   | 360 B  (x‖y‖z, each 120 B)          |
//! | G2      | 192 B uncompressed        | 720 B  (x.c0‖x.c1‖y.c0‖y.c1‖z.c0‖z.c1) |
//! | Fr      | 32 B, little-endian       | 32 B, little-endian (8 × u32)        |

use std::ops::{Add, Mul, Sub};

use ff::{Field, PrimeField, PrimeFieldBits};
use group::Group;
use group::prime::PrimeCurveAffine;

/// Size of a single Fq coordinate in standard (Zcash) form: 48 bytes (384 bits, big-endian).
const FQ_COORD_SIZE: usize = 48;

/// Mask to strip the top 3 Zcash metadata bits (compression, infinity, sign flags)
/// from the first byte of an uncompressed point serialization.
const ZCASH_METADATA_MASK: u8 = 0b0001_1111;

/// Size of a single Fq element in GPU 13-bit limb format: 30 limbs × 4 bytes = 120 bytes.
pub const FQ_GPU_BYTES: usize = 120;
/// Size of a G1 point in GPU format (Jacobian: x, y, z): 3 × 120 = 360 bytes.
pub const G1_GPU_BYTES: usize = 3 * FQ_GPU_BYTES;
/// Size of a G2 point in GPU format (Jacobian over Fq2: x.c0, x.c1, y.c0, y.c1, z.c0, z.c1):
/// 6 × 120 = 720 bytes.
pub const G2_GPU_BYTES: usize = 6 * FQ_GPU_BYTES;

/// Convert a 48-byte **little-endian** Fq element to the GPU's 30×13-bit limb format (120 bytes).
///
/// The GPU uses R = 2^390 Montgomery representation with 30 limbs of 13 bits each
/// (instead of the more common 12×32-bit / R = 2^384 layout). This avoids expensive
/// `mul_u32` 16-bit decomposition in WGSL — native u32 × u32 products stay within
/// 26 bits and don't overflow.
///
/// Each 13-bit limb is zero-extended to a full `u32` and stored as 4 bytes LE.
/// The bit-packing walks a sliding 13-bit window across the input bytes.
pub(crate) fn fq_bytes_to_13bit(bytes_48: &[u8]) -> Vec<u8> {
    debug_assert_eq!(bytes_48.len(), 48);
    let mut result = vec![0u8; FQ_GPU_BYTES];
    let mut bit_offset: usize = 0;
    for i in 0..30 {
        let byte_idx = bit_offset / 8;
        let bit_shift = bit_offset % 8;
        let mut val: u32 = 0;
        for j in 0..3usize {
            if byte_idx + j < 48 {
                val |= (bytes_48[byte_idx + j] as u32) << (j * 8);
            }
        }
        let limb = (val >> bit_shift) & 0x1FFF;
        let limb_bytes = limb.to_le_bytes();
        result[i * 4..i * 4 + 4].copy_from_slice(&limb_bytes);
        bit_offset += 13;
    }
    result
}

/// Inverse of [`fq_bytes_to_13bit`]: convert 120 GPU limb bytes back to 48-byte LE Fq.
///
/// Each 4-byte chunk is read as a u32 LE and masked to 13 bits, then the bits are
/// scattered back into the packed 384-bit output.
pub(crate) fn fq_13bit_to_bytes(bytes_120: &[u8]) -> Vec<u8> {
    debug_assert_eq!(bytes_120.len(), FQ_GPU_BYTES);
    let mut result = vec![0u8; 48];
    let mut bit_offset: usize = 0;
    for i in 0..30 {
        let limb = u32::from_le_bytes([
            bytes_120[i * 4],
            bytes_120[i * 4 + 1],
            bytes_120[i * 4 + 2],
            bytes_120[i * 4 + 3],
        ]) & 0x1FFF;
        let byte_idx = bit_offset / 8;
        let bit_shift = bit_offset % 8;
        let shifted = (limb as u64) << bit_shift;
        for j in 0..3usize {
            if byte_idx + j < 48 {
                result[byte_idx + j] |= ((shifted >> (j * 8)) & 0xFF) as u8;
            }
        }
        bit_offset += 13;
    }
    result
}

/// Convenience: big-endian Fq coordinate (as stored in Zcash uncompressed format) → GPU limbs.
///
/// Equivalent to reversing to LE then calling [`fq_bytes_to_13bit`].
fn be_coord_to_gpu_limbs(be_bytes: &[u8]) -> Vec<u8> {
    let mut le = be_bytes.to_vec();
    le.reverse();
    fq_bytes_to_13bit(&le)
}

/// Inverse of [`be_coord_to_gpu_limbs`]: GPU limbs → big-endian Fq coordinate.
fn gpu_limbs_to_be_coord(limb_bytes: &[u8]) -> Vec<u8> {
    let mut be = fq_13bit_to_bytes(limb_bytes);
    be.reverse();
    be
}

/// Returns true if all bytes are zero (used to check Z coordinate for point at infinity).
fn is_all_zero(bytes: &[u8]) -> bool {
    bytes.iter().all(|&b| b == 0)
}

/// Abstraction over a pairing-friendly curve for GPU-accelerated proving.
///
/// Provides the serialization bridge between CPU curve libraries (blstrs) and GPU
/// shader buffers, scalar decomposition for Pippenger MSM, NTT root-of-unity
/// computation, and basic point arithmetic used in CPU-side folding/verification.
///
/// Currently implemented only for [`blstrs::Bls12`] (BLS12-381).
pub trait GpuCurve: 'static {
    type Engine: pairing::Engine;

    type Scalar: PrimeField + PrimeFieldBits;
    type G1: Group<Scalar = Self::Scalar>;
    type G2: Group<Scalar = Self::Scalar>;
    type G1Affine;
    type G2Affine;

    /// WGSL source for NTT kernels (Fr field + Fp field + NTT entry points).
    const NTT_SOURCE: &'static str;
    /// WGSL source for fused NTT+coset-shift kernel (NTT source + fused entry point).
    const NTT_FUSED_SOURCE: &'static str;
    /// WGSL source for MSM kernels (Fr + Fp + curve arithmetic + MSM entry points).
    const MSM_SOURCE: &'static str;
    /// WGSL source for polynomial-evaluation kernels (Fr + Fp + poly_ops).
    const POLY_OPS_SOURCE: &'static str;

    /// Serialize an affine G1 point to GPU buffer format (360 bytes).
    ///
    /// Converts from Zcash uncompressed (96 bytes, BE x‖y) to Jacobian 13-bit limbs
    /// (x‖y‖z, each 120 bytes) with Z = 1. Identity maps to all-zeros.
    fn serialize_g1(point: &Self::G1Affine) -> Vec<u8>;
    /// Serialize an affine G2 point to GPU buffer format (720 bytes).
    ///
    /// Converts from Zcash uncompressed (192 bytes) to Jacobian-over-Fq2 13-bit limbs.
    /// Note the component reordering: Zcash stores `(x.c1, x.c0, y.c1, y.c0)` while
    /// the GPU layout is `(x.c0, x.c1, y.c0, y.c1, z.c0, z.c1)`.
    fn serialize_g2(point: &Self::G2Affine) -> Vec<u8>;
    /// Serialize a scalar field element to GPU format (32 bytes, LE).
    fn serialize_scalar(s: &Self::Scalar) -> Vec<u8>;
    /// Deserialize a scalar from 32-byte LE GPU buffer.
    fn deserialize_scalar(bytes: &[u8]) -> anyhow::Result<Self::Scalar>;
    /// Deserialize a G1 point from 360-byte GPU buffer back to projective form.
    ///
    /// Checks the Z coordinate for all-zeros to detect the point at infinity.
    fn deserialize_g1(bytes: &[u8]) -> anyhow::Result<Self::G1>;
    /// Deserialize a G2 point from 720-byte GPU buffer back to projective form.
    fn deserialize_g2(bytes: &[u8]) -> anyhow::Result<Self::G2>;

    /// Unsigned scalar window decomposition for Pippenger bucket sort.
    ///
    /// Splits a 256-bit scalar into `ceil(256/c)` windows of `c` bits each.
    /// Each window value is in `[0, 2^c)`.
    fn scalar_to_windows(s: &Self::Scalar, c: usize) -> Vec<u32>;

    /// Signed-digit scalar decomposition for Pippenger with halved bucket count.
    ///
    /// Each window value is in `[-(2^(c-1)), 2^(c-1)]`, returned as
    /// `(absolute_value, is_negative)` pairs. This halves the number of buckets
    /// compared to unsigned decomposition (2^(c-1) instead of 2^c), at the cost
    /// of negating the corresponding base point on the GPU for negative windows.
    /// A carry propagates upward when a window value reaches the threshold 2^(c-1).
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
                // Negative window: value = -(2^c - val), carry +1 to next window
                let abs = full - val;
                result.push((abs as u32, true));
                carry = 1;
            } else {
                result.push((val as u32, false));
            }
        }

        // Handle final carry (extra window needed)
        if carry > 0 {
            result.push((1, false));
        }

        result
    }

    /// Optimal bucket width `c` for G1 MSM (Pippenger). The MSM uses 2^(c-1)
    /// buckets per window (with signed decomposition). Larger `c` means fewer
    /// windows but more buckets — the sweet spot for BLS12-381 G1 is c=13.
    fn bucket_width() -> usize;

    /// Bucket width for G2 MSM. Smaller than G1 because G2 point operations are
    /// ~3× more expensive (Fq2 vs Fq). G2 now uses parallel tree reduction (like G1)
    /// via add_g2_complete, so c=10/12 are viable but show negligible improvement
    /// at current point set sizes (~21K).
    fn g2_bucket_width() -> usize {
        8
    }

    /// Bucket width for GLV-accelerated G1 MSM (~128-bit sub-scalars).
    /// With GLV, each 255-bit scalar is decomposed into two ~128-bit sub-scalars,
    /// halving the number of Pippenger windows.
    fn glv_bucket_width() -> usize {
        13
    }

    /// Compute a primitive 2^n-th root of unity in the scalar field Fr.
    /// Used for NTT (Number Theoretic Transform) in H-polynomial evaluation.
    fn root_of_unity(n: usize) -> Self::Scalar;

    /// G1 additive identity (point at infinity).
    fn g1_identity() -> Self::G1 {
        Self::G1::identity()
    }
    /// G2 additive identity (point at infinity).
    fn g2_identity() -> Self::G2 {
        Self::G2::identity()
    }

    // --- Point conversion and arithmetic (used in CPU-side window-sum folding) ---

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

/// BLS12-381 implementation using the `blstrs` crate.
///
/// Shader source is assembled at compile time by concatenating WGSL includes.
/// The concatenation order matters — later files reference types/functions from
/// earlier ones (e.g. `msm.wgsl` depends on `curve.wgsl` depends on `fp.wgsl`).
impl GpuCurve for blstrs::Bls12 {
    type Engine = Self;
    type Scalar = <Self::Engine as pairing::Engine>::Fr;
    type G1 = <Self::Engine as pairing::Engine>::G1;
    type G2 = <Self::Engine as pairing::Engine>::G2;
    type G1Affine = <Self::Engine as pairing::Engine>::G1Affine;
    type G2Affine = <Self::Engine as pairing::Engine>::G2Affine;

    const NTT_SOURCE: &'static str = concat!(
        include_str!("../shader/bls12_381/fr.wgsl"),
        "\n",
        include_str!("../shader/bls12_381/fp.wgsl"),
        "\n",
        include_str!("../shader/bls12_381/ntt.wgsl"),
    );

    const NTT_FUSED_SOURCE: &'static str = concat!(
        include_str!("../shader/bls12_381/fr.wgsl"),
        "\n",
        include_str!("../shader/bls12_381/fp.wgsl"),
        "\n",
        include_str!("../shader/bls12_381/ntt.wgsl"),
        "\n",
        include_str!("../shader/bls12_381/ntt_fused.wgsl"),
    );

    const MSM_SOURCE: &'static str = concat!(
        include_str!("../shader/bls12_381/fr.wgsl"),
        "\n",
        include_str!("../shader/bls12_381/fp.wgsl"),
        "\n",
        include_str!("../shader/bls12_381/curve.wgsl"),
        "\n",
        include_str!("../shader/bls12_381/msm.wgsl"),
    );

    const POLY_OPS_SOURCE: &'static str = concat!(
        include_str!("../shader/bls12_381/fr.wgsl"),
        "\n",
        include_str!("../shader/bls12_381/fp.wgsl"),
        "\n",
        include_str!("../shader/bls12_381/poly_ops.wgsl"),
    );

    /// Serialize affine G1 → 360-byte GPU Jacobian.
    ///
    /// Layout: `[x: 120B] [y: 120B] [z: 120B]` where each coordinate is
    /// 30×13-bit limbs (u32 LE). Z = 1 (not in Montgomery form — the GPU's
    /// `to_montgomery_bases` kernel converts later).
    fn serialize_g1(point: &Self::G1Affine) -> Vec<u8> {
        let is_inf: bool = point.is_identity().into();
        if is_inf {
            return vec![0u8; G1_GPU_BYTES];
        }

        let mut uncompressed = point.to_uncompressed();
        uncompressed[0] &= ZCASH_METADATA_MASK;

        // z = 1 in standard form (not Montgomery — GPU converts to Montgomery later)
        let mut z_le = vec![0u8; FQ_COORD_SIZE];
        z_le[0] = 1;

        let mut wgsl_bytes = Vec::with_capacity(G1_GPU_BYTES);
        wgsl_bytes.extend_from_slice(&be_coord_to_gpu_limbs(&uncompressed[..FQ_COORD_SIZE]));
        wgsl_bytes.extend_from_slice(&be_coord_to_gpu_limbs(&uncompressed[FQ_COORD_SIZE..2 * FQ_COORD_SIZE]));
        wgsl_bytes.extend_from_slice(&fq_bytes_to_13bit(&z_le));
        wgsl_bytes
    }

    /// Serialize affine G2 → 720-byte GPU Jacobian-over-Fq2.
    ///
    /// Layout: `[x.c0][x.c1][y.c0][y.c1][z.c0][z.c1]`, each 120 bytes.
    /// Note the Zcash ↔ GPU component reordering: blstrs stores
    /// `(x.c1, x.c0, y.c1, y.c0)` while the GPU expects `(c0, c1)` order.
    fn serialize_g2(point: &Self::G2Affine) -> Vec<u8> {
        let is_inf: bool = point.is_identity().into();
        if is_inf {
            return vec![0u8; G2_GPU_BYTES];
        }

        let mut uncompressed = point.to_uncompressed();
        uncompressed[0] &= ZCASH_METADATA_MASK;

        // blstrs G2 uncompressed layout (192 bytes):
        // x.c1 (48B BE), x.c0 (48B BE), y.c1 (48B BE), y.c0 (48B BE)
        let s = FQ_COORD_SIZE;
        let x_c1 = be_coord_to_gpu_limbs(&uncompressed[0..s]);
        let x_c0 = be_coord_to_gpu_limbs(&uncompressed[s..2 * s]);
        let y_c1 = be_coord_to_gpu_limbs(&uncompressed[2 * s..3 * s]);
        let y_c0 = be_coord_to_gpu_limbs(&uncompressed[3 * s..4 * s]);

        // z = (1, 0) in standard form
        let mut z_c0_le = vec![0u8; FQ_COORD_SIZE];
        z_c0_le[0] = 1;
        let z_c1_le = vec![0u8; FQ_COORD_SIZE];

        // GPU layout: x.c0, x.c1, y.c0, y.c1, z.c0, z.c1
        let mut wgsl_bytes = Vec::with_capacity(G2_GPU_BYTES);
        wgsl_bytes.extend_from_slice(&x_c0);
        wgsl_bytes.extend_from_slice(&x_c1);
        wgsl_bytes.extend_from_slice(&y_c0);
        wgsl_bytes.extend_from_slice(&y_c1);
        wgsl_bytes.extend_from_slice(&fq_bytes_to_13bit(&z_c0_le));
        wgsl_bytes.extend_from_slice(&fq_bytes_to_13bit(&z_c1_le));
        wgsl_bytes
    }

    /// Deserialize 360-byte GPU Jacobian → projective G1.
    ///
    /// The GPU writes results in standard (non-Montgomery) affine form, so only
    /// x and y need to be recovered. Z = 0 signals the point at infinity.
    fn deserialize_g1(bytes: &[u8]) -> anyhow::Result<Self::G1> {
        if bytes.len() != G1_GPU_BYTES {
            anyhow::bail!("Invalid G1 byte length from GPU: {}", bytes.len());
        }
        if is_all_zero(&bytes[2 * FQ_GPU_BYTES..3 * FQ_GPU_BYTES]) {
            return Ok(Self::G1::identity());
        }

        let x_be = gpu_limbs_to_be_coord(&bytes[0..FQ_GPU_BYTES]);
        let y_be = gpu_limbs_to_be_coord(&bytes[FQ_GPU_BYTES..2 * FQ_GPU_BYTES]);

        let mut uncompressed = [0u8; 96];
        uncompressed[..FQ_COORD_SIZE].copy_from_slice(&x_be);
        uncompressed[FQ_COORD_SIZE..].copy_from_slice(&y_be);

        let ct: subtle::CtOption<blstrs::G1Affine> =
            blstrs::G1Affine::from_uncompressed(&uncompressed);
        let affine: Option<blstrs::G1Affine> = ct.into();

        if let Some(affine) = affine {
            Ok(affine.into())
        } else {
            let limb0 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            anyhow::bail!(
                "Failed to deserialize G1 point from GPU (x_limb0={:#x})",
                limb0
            )
        }
    }

    /// Deserialize 720-byte GPU Jacobian-over-Fq2 → projective G2.
    ///
    /// Reverses the component reordering from [`serialize_g2`] back to Zcash order
    /// before calling `from_uncompressed`.
    fn deserialize_g2(bytes: &[u8]) -> anyhow::Result<Self::G2> {
        if bytes.len() != G2_GPU_BYTES {
            anyhow::bail!("Invalid G2 byte length from GPU: {}", bytes.len());
        }
        if is_all_zero(&bytes[4 * FQ_GPU_BYTES..6 * FQ_GPU_BYTES]) {
            return Ok(Self::G2::identity());
        }

        let x_c0_be = gpu_limbs_to_be_coord(&bytes[0..FQ_GPU_BYTES]);
        let x_c1_be = gpu_limbs_to_be_coord(&bytes[FQ_GPU_BYTES..2 * FQ_GPU_BYTES]);
        let y_c0_be = gpu_limbs_to_be_coord(&bytes[2 * FQ_GPU_BYTES..3 * FQ_GPU_BYTES]);
        let y_c1_be = gpu_limbs_to_be_coord(&bytes[3 * FQ_GPU_BYTES..4 * FQ_GPU_BYTES]);

        // Reconstruct Zcash BE uncompressed structure: x_c1, x_c0, y_c1, y_c0
        let s = FQ_COORD_SIZE;
        let mut uncompressed = [0u8; 192];
        uncompressed[0..s].copy_from_slice(&x_c1_be);
        uncompressed[s..2 * s].copy_from_slice(&x_c0_be);
        uncompressed[2 * s..3 * s].copy_from_slice(&y_c1_be);
        uncompressed[3 * s..4 * s].copy_from_slice(&y_c0_be);

        let ct: subtle::CtOption<blstrs::G2Affine> =
            blstrs::G2Affine::from_uncompressed(&uncompressed);
        let affine: Option<blstrs::G2Affine> = ct.into();

        if let Some(affine) = affine {
            Ok(affine.into())
        } else {
            anyhow::bail!("Failed to deserialize G2 point from GPU")
        }
    }

    /// Serialize Fr scalar → 32-byte LE buffer for GPU upload.
    fn serialize_scalar(s: &Self::Scalar) -> Vec<u8> {
        let bits = s.to_le_bits();
        let mut bytes = [0u8; 32];
        for (i, bit) in bits.iter().enumerate() {
            if *bit {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }
        bytes.to_vec()
    }

    fn deserialize_scalar(bytes: &[u8]) -> anyhow::Result<Self::Scalar> {
        let mut arr = [0u8; 32];
        arr.copy_from_slice(bytes);
        let scalar: Option<blstrs::Scalar> = blstrs::Scalar::from_bytes_le(&arr).into();
        scalar.ok_or_else(|| anyhow::anyhow!("Invalid scalar bytes from GPU"))
    }

    /// Extract `ceil(256/c)` unsigned windows of `c` bits each from the scalar.
    ///
    /// Walks a sliding `c`-bit window across the 32-byte LE representation,
    /// handling cross-byte boundaries. Used by Pippenger bucket sort to assign
    /// each scalar to one bucket per window.
    fn scalar_to_windows(s: &Self::Scalar, c: usize) -> Vec<u32> {
        let bits = s.to_le_bits();
        let mut bytes = [0u8; 32];
        for (i, bit) in bits.iter().enumerate() {
            if *bit {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }

        let num_windows = 256_usize.div_ceil(c);
        let mut windows = Vec::with_capacity(num_windows);

        for i in 0..num_windows {
            let bit_offset = i * c;
            let mut window: u64 = 0;
            let bit_shift = bit_offset % 8;
            let bytes_needed = (bit_shift + c).div_ceil(8).min(8);

            for j in 0..bytes_needed {
                let byte_idx = bit_offset / 8 + j;
                if byte_idx < 32 {
                    window |= (bytes[byte_idx] as u64) << (j * 8);
                }
            }

            let mask = (1u64 << c) - 1;
            windows.push(((window >> bit_shift) & mask) as u32);
        }

        windows
    }

    fn bucket_width() -> usize {
        13
    }

    fn g1_identity() -> Self::G1 {
        Self::G1::identity()
    }

    /// Compute a primitive 2^n-th root of unity by exponentiating the 2^32-th
    /// root of unity. BLS12-381 Fr has a 2-adicity of 32, so n must be ≤ 2^32.
    fn root_of_unity(n: usize) -> Self::Scalar {
        let exponent = 0x100000000u64 >> n.trailing_zeros();
        blstrs::Scalar::ROOT_OF_UNITY.pow_vartime([exponent])
    }

    fn affine_to_proj_g1(p: &Self::G1Affine) -> Self::G1 {
        p.into()
    }
    fn affine_to_proj_g2(p: &Self::G2Affine) -> Self::G2 {
        p.into()
    }
    fn proj_to_affine_g1(p: &Self::G1) -> Self::G1Affine {
        p.into()
    }
    fn proj_to_affine_g2(p: &Self::G2) -> Self::G2Affine {
        p.into()
    }

    fn add_g1_proj(a: &Self::G1, b: &Self::G1) -> Self::G1 {
        a.add(b)
    }
    fn sub_g1_proj(a: &Self::G1, b: &Self::G1) -> Self::G1 {
        a.sub(b)
    }
    fn add_g2_proj(a: &Self::G2, b: &Self::G2) -> Self::G2 {
        a.add(b)
    }

    fn mul_g1_scalar(a: &Self::G1Affine, b: &Self::Scalar) -> Self::G1 {
        a.mul(b)
    }
    fn mul_g2_scalar(a: &Self::G2Affine, b: &Self::Scalar) -> Self::G2 {
        a.mul(b)
    }
    fn mul_g1_proj_scalar(a: &Self::G1, b: &Self::Scalar) -> Self::G1 {
        a.mul(b)
    }

    fn g2_identity() -> Self::G2 {
        Self::G2::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuCurve, FQ_GPU_BYTES, G1_GPU_BYTES, G2_GPU_BYTES};
    use blstrs::{Bls12, G1Affine, G2Affine, Scalar};
    use ff::{PrimeField, PrimeFieldBits};
    use group::Group;
    use group::prime::PrimeCurveAffine;

    #[test]
    fn g1_serialize_deserialize_round_trip() {
        let point = G1Affine::generator();
        let bytes = <Bls12 as GpuCurve>::serialize_g1(&point);
        let parsed = <Bls12 as GpuCurve>::deserialize_g1(&bytes).expect("g1 deserialize failed");
        let parsed_affine: G1Affine = parsed.into();
        assert_eq!(parsed_affine, point);
    }

    #[test]
    fn g2_serialize_deserialize_round_trip() {
        let point = G2Affine::generator();
        let bytes = <Bls12 as GpuCurve>::serialize_g2(&point);
        let parsed = <Bls12 as GpuCurve>::deserialize_g2(&bytes).expect("g2 deserialize failed");
        let parsed_affine: G2Affine = parsed.into();
        assert_eq!(parsed_affine, point);
    }

    fn expected_windows(s: &Scalar, c: usize) -> Vec<u32> {
        let bits = s.to_le_bits();
        let num_windows = 256_usize.div_ceil(c);
        let mut out = Vec::with_capacity(num_windows);
        for i in 0..num_windows {
            let bit_offset = i * c;
            let mut w = 0u32;
            for j in 0..c {
                let idx = bit_offset + j;
                if idx < 256 && bits[idx] {
                    w |= 1u32 << j;
                }
            }
            out.push(w);
        }
        out
    }

    #[test]
    fn scalar_window_decomposition_matches_bit_extraction() {
        let c = <Bls12 as GpuCurve>::bucket_width();
        let samples = [
            Scalar::from(0u64),
            Scalar::from(1u64),
            Scalar::from(2u64),
            Scalar::from(3u64),
            Scalar::from(0x1234_5678_9abc_def0u64),
            -Scalar::from(5u64),
            Scalar::ROOT_OF_UNITY,
        ];

        for s in samples {
            let got = <Bls12 as GpuCurve>::scalar_to_windows(&s, c);
            let exp = expected_windows(&s, c);
            assert_eq!(got, exp);
        }
    }

    /// Verify that signed windows reconstruct the original scalar.
    #[test]
    fn signed_window_decomposition_roundtrip() {
        use ff::Field;
        use rand_core::OsRng;

        let c = <Bls12 as GpuCurve>::bucket_width();
        let samples = [
            Scalar::from(0u64),
            Scalar::from(1u64),
            Scalar::from(2u64),
            Scalar::from(0x1234_5678_9abc_def0u64),
            -Scalar::from(5u64),
            Scalar::ROOT_OF_UNITY,
            Scalar::random(OsRng),
            Scalar::random(OsRng),
            Scalar::random(OsRng),
        ];

        let base = Scalar::from(1u64 << c);
        let half = 1u32 << (c - 1);

        for s in samples {
            let signed = <Bls12 as GpuCurve>::scalar_to_signed_windows(&s, c);

            // Reconstruct scalar from signed windows: ∑ (±abs) * 2^(i*c)
            let mut reconstructed = Scalar::ZERO;
            let mut power = Scalar::ONE;
            for &(abs, neg) in &signed {
                let term = Scalar::from(abs as u64) * power;
                if neg {
                    reconstructed -= term;
                } else {
                    reconstructed += term;
                }
                power *= base;
            }

            assert_eq!(
                reconstructed, s,
                "signed windows must reconstruct original scalar"
            );

            // Verify all absolute values are in [0, 2^(c-1)]
            for &(abs, _neg) in &signed {
                assert!(abs <= half, "abs value {} exceeds 2^(c-1) = {}", abs, half);
            }
        }
    }

    /// Verify signed decomposition handles the exact half-boundary correctly.
    /// Scalar = 2^(c-1) should produce window 0 = (2^(c-1), true) with carry.
    #[test]
    fn signed_window_decomposition_half_boundary() {
        let c = <Bls12 as GpuCurve>::bucket_width();
        let half = 1u64 << (c - 1);
        let s = Scalar::from(half);
        let signed = <Bls12 as GpuCurve>::scalar_to_signed_windows(&s, c);

        // Window 0: val=half is >= half, so abs = 2^c - half = half, neg=true, carry=1
        assert_eq!(signed[0], (half as u32, true));
        // Window 1: carry=1, which is < half, so (1, false)
        assert_eq!(signed[1], (1, false));
        // Remaining windows should be (0, false)
        for &(abs, neg) in &signed[2..] {
            assert_eq!((abs, neg), (0, false));
        }
    }

    /// Verify that scalar = 2^(c-1) - 1 stays positive (just below the threshold).
    #[test]
    fn signed_window_decomposition_below_half() {
        let c = <Bls12 as GpuCurve>::bucket_width();
        let val = (1u64 << (c - 1)) - 1;
        let s = Scalar::from(val);
        let signed = <Bls12 as GpuCurve>::scalar_to_signed_windows(&s, c);

        // Window 0: val < half, so (val, false), no carry
        assert_eq!(signed[0], (val as u32, false));
        // All remaining windows should be zero
        for &(abs, _) in &signed[1..] {
            assert_eq!(abs, 0);
        }
    }

    /// Serializing the G1 identity (point at infinity) and deserializing it
    /// must round-trip to the identity element.
    #[test]
    fn g1_identity_serialize_deserialize() {
        let identity = G1Affine::identity();
        let bytes = <Bls12 as GpuCurve>::serialize_g1(&identity);
        assert_eq!(bytes.len(), G1_GPU_BYTES);

        // All bytes should be zero for identity
        assert!(bytes.iter().all(|&b| b == 0));

        let parsed = <Bls12 as GpuCurve>::deserialize_g1(&bytes).expect("identity deserialize");
        assert!(
            bool::from(parsed.is_identity()),
            "deserialized identity should be identity"
        );
    }

    /// Serializing the G2 identity and deserializing it must round-trip.
    #[test]
    fn g2_identity_serialize_deserialize() {
        let identity = G2Affine::identity();
        let bytes = <Bls12 as GpuCurve>::serialize_g2(&identity);
        assert_eq!(bytes.len(), G2_GPU_BYTES);

        // All bytes should be zero for identity
        assert!(bytes.iter().all(|&b| b == 0));

        let parsed = <Bls12 as GpuCurve>::deserialize_g2(&bytes).expect("g2 identity deserialize");
        assert!(
            bool::from(parsed.is_identity()),
            "deserialized G2 identity should be identity"
        );
    }

    /// Multiple random G1 points round-trip through serialize/deserialize.
    #[test]
    fn g1_random_points_roundtrip() {
        use group::Group;

        let g = blstrs::G1Projective::generator();
        for i in 1..20u64 {
            let point = g * Scalar::from(i);
            let affine: G1Affine = point.into();
            let bytes = <Bls12 as GpuCurve>::serialize_g1(&affine);
            let parsed =
                <Bls12 as GpuCurve>::deserialize_g1(&bytes).expect("random g1 deserialize");
            let parsed_affine: G1Affine = parsed.into();
            assert_eq!(
                parsed_affine, affine,
                "G1 roundtrip failed for scalar multiplier {i}"
            );
        }
    }

    /// Multiple random G2 points round-trip through serialize/deserialize.
    #[test]
    fn g2_random_points_roundtrip() {
        use group::Group;

        let g = blstrs::G2Projective::generator();
        for i in 1..10u64 {
            let point = g * Scalar::from(i);
            let affine: G2Affine = point.into();
            let bytes = <Bls12 as GpuCurve>::serialize_g2(&affine);
            let parsed =
                <Bls12 as GpuCurve>::deserialize_g2(&bytes).expect("random g2 deserialize");
            let parsed_affine: G2Affine = parsed.into();
            assert_eq!(
                parsed_affine, affine,
                "G2 roundtrip failed for scalar multiplier {i}"
            );
        }
    }

    /// Scalar serialization round-trips for special values.
    #[test]
    fn scalar_serialize_deserialize_roundtrip() {
        use ff::Field;
        use rand_core::OsRng;

        let test_scalars = [
            Scalar::ZERO,
            Scalar::ONE,
            Scalar::from(2u64),
            Scalar::from(0xFFFF_FFFF_FFFF_FFFFu64),
            -Scalar::ONE,
            Scalar::ROOT_OF_UNITY,
            Scalar::random(OsRng),
            Scalar::random(OsRng),
        ];

        for (i, s) in test_scalars.iter().enumerate() {
            let bytes = <Bls12 as GpuCurve>::serialize_scalar(s);
            assert_eq!(bytes.len(), 32, "scalar bytes length should be 32");
            let parsed =
                <Bls12 as GpuCurve>::deserialize_scalar(&bytes).expect("scalar deserialize");
            assert_eq!(parsed, *s, "scalar roundtrip failed for test case {i}");
        }
    }

    /// G1 serialization byte layout: x(48 LE) || y(48 LE) || z(48 LE).
    #[test]
    fn g1_serialization_byte_layout() {
        let g = G1Affine::generator();
        let bytes = <Bls12 as GpuCurve>::serialize_g1(&g);

        assert_eq!(bytes.len(), G1_GPU_BYTES);

        // x and y should not be all zeros for the generator
        assert!(!bytes[0..FQ_GPU_BYTES].iter().all(|&b| b == 0));
        assert!(!bytes[FQ_GPU_BYTES..2 * FQ_GPU_BYTES].iter().all(|&b| b == 0));

        // z = 1 in 13-bit format: limb[0] = 1 (bytes [0..4] = [1,0,0,0]), rest zeros
        let z_start = 2 * FQ_GPU_BYTES;
        assert_eq!(bytes[z_start], 1);
        assert!(bytes[z_start + 1..G1_GPU_BYTES].iter().all(|&b| b == 0));
    }

    /// G1 deserialization rejects wrong-length input.
    #[test]
    fn g1_deserialize_rejects_wrong_length() {
        let short = vec![0u8; 100];
        assert!(<Bls12 as GpuCurve>::deserialize_g1(&short).is_err());

        let long = vec![0u8; 200];
        assert!(<Bls12 as GpuCurve>::deserialize_g1(&long).is_err());
    }

    /// Unsigned window decomposition: all windows are < 2^c.
    #[test]
    fn scalar_window_values_bounded() {
        use ff::Field;
        use rand_core::OsRng;

        let c = <Bls12 as GpuCurve>::bucket_width();
        let max_val = (1u64 << c) - 1;

        for _ in 0..50 {
            let s = Scalar::random(OsRng);
            let windows = <Bls12 as GpuCurve>::scalar_to_windows(&s, c);
            for (i, &w) in windows.iter().enumerate() {
                assert!(
                    (w as u64) <= max_val,
                    "window {i} value {w} exceeds max {max_val}"
                );
            }
        }
    }

    /// Window decomposition with different widths (c=8, c=13, c=16) always reconstructs.
    #[test]
    fn scalar_window_decomposition_various_widths() {
        use ff::Field;
        use rand_core::OsRng;

        for c in [8, 13, 16] {
            let base = Scalar::from(1u64 << c);

            for _ in 0..20 {
                let s = Scalar::random(OsRng);
                let windows = <Bls12 as GpuCurve>::scalar_to_windows(&s, c);

                let mut reconstructed = Scalar::ZERO;
                let mut power = Scalar::ONE;
                for &w in &windows {
                    reconstructed += Scalar::from(w as u64) * power;
                    power *= base;
                }
                assert_eq!(reconstructed, s, "unsigned decomposition failed for c={c}");
            }
        }
    }

    /// Signed window decomposition with various widths always reconstructs.
    #[test]
    fn signed_window_decomposition_various_widths() {
        use ff::Field;
        use rand_core::OsRng;

        for c in [8, 13, 16] {
            let base = Scalar::from(1u64 << c);
            let half = 1u32 << (c - 1);

            for _ in 0..20 {
                let s = Scalar::random(OsRng);
                let signed = <Bls12 as GpuCurve>::scalar_to_signed_windows(&s, c);

                let mut reconstructed = Scalar::ZERO;
                let mut power = Scalar::ONE;
                for &(abs, neg) in &signed {
                    assert!(abs <= half, "abs {abs} > half {half} for c={c}");
                    let term = Scalar::from(abs as u64) * power;
                    if neg {
                        reconstructed -= term;
                    } else {
                        reconstructed += term;
                    }
                    power *= base;
                }
                assert_eq!(reconstructed, s, "signed decomposition failed for c={c}");
            }
        }
    }

    /// Verify signed windows for G2 bucket width.
    #[test]
    fn signed_window_decomposition_g2_roundtrip() {
        use ff::Field;
        use rand_core::OsRng;

        let c = <Bls12 as GpuCurve>::g2_bucket_width();
        let base = Scalar::from(1u64 << c);
        let half = 1u32 << (c - 1);

        for _ in 0..20 {
            let s = Scalar::random(OsRng);
            let signed = <Bls12 as GpuCurve>::scalar_to_signed_windows(&s, c);

            let mut reconstructed = Scalar::ZERO;
            let mut power = Scalar::ONE;
            for &(abs, neg) in &signed {
                let term = Scalar::from(abs as u64) * power;
                if neg {
                    reconstructed -= term;
                } else {
                    reconstructed += term;
                }
                power *= base;
            }

            assert_eq!(reconstructed, s);
            for &(abs, _) in &signed {
                assert!(abs <= half);
            }
        }
    }
}
