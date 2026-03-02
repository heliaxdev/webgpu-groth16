//! BLS12-381 GLV (Gallant-Lambert-Vanstone) endomorphism optimization for G1.
//!
//! Exploits the efficient endomorphism φ(x, y) = (β·x, y) on BLS12-381 G1 to
//! decompose any 256-bit scalar k into two ~128-bit scalars k1, k2 such that
//! k·P = k1·P + k2·φ(P). This halves the number of Pippenger windows in MSM.
//!
//! ## Decomposition algorithm (Babai rounding)
//!
//! The BLS12-381 scalar field r has a short lattice basis derived from the
//! curve parameter x = -0xd201000000010000:
//!
//!   v1 = (N11, 1)     where N11 = x²
//!   v2 = (N22, -1)    where N22 = x² - 1
//!
//! Given scalar k, we compute:
//!   1. c1 = round(k/r): since 0 ≤ k < r, this is 0 or 1
//!   2. c2 = round(k·N22/r) via precomputed g = round(N22·2^256/r)
//!   3. k1 = k - c1 - c2·N11
//!   4. k2 = c1·N22 - c2
//!
//! Both |k1| and |k2| are bounded by ~sqrt(r) ≈ 2^128.

use blstrs::{Fp, G1Affine};
use ff::PrimeField;
use group::prime::PrimeCurveAffine;

use crate::gpu::curve::GpuCurve;
use crate::gpu::curve::bls12_381::{fq_13bit_to_bytes, fq_bytes_to_13bit};

const FQ_GPU_BYTES: usize = <blstrs::Bls12 as GpuCurve>::FQ_GPU_BYTES;
const FQ_GPU_PADDED_BYTES: usize =
    <blstrs::Bls12 as GpuCurve>::FQ_GPU_PADDED_BYTES;
const G1_GPU_BYTES: usize = <blstrs::Bls12 as GpuCurve>::G1_GPU_BYTES;

// ============================================================================
// BLS12-381 GLV constants
// ============================================================================

/// Cube root of unity β in Fq such that φ(x,y) = (β·x, y) is an endomorphism.
/// β^3 = 1 (mod q), β ≠ 1. Equivalently: β = g^((q-1)/3) where g generates Fq*.
/// Reference: https://eprint.iacr.org/2013/158 Section 4
const BETA_LE_BYTES: [u8; 48] = [
    0xfe, 0xff, 0xfe, 0xff, 0xff, 0xff, 0x01, 0x2e, 0x02, 0x00, 0x0a, 0x62,
    0x13, 0xd8, 0x17, 0xde, 0x88, 0x96, 0xf8, 0xe6, 0x3b, 0xa9, 0xb3, 0xdd,
    0xea, 0x77, 0x0f, 0x6a, 0x07, 0xc6, 0x69, 0xba, 0x51, 0xce, 0x76, 0xdf,
    0x2f, 0x67, 0x19, 0x5f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// Lattice vector component N11 = x² where x = -0xd201000000010000 (BLS
/// parameter). N11 = 228988810152649578064853576960394133504 =
/// 0xac45a4010001a4020000000100000000
const N11_LO: u64 = 0x0000000100000000;
const N11_HI: u64 = 0xac45a4010001a402;

/// Lattice vector component N22 = x² - 1.
/// N22 = 228988810152649578064853576960394133503 =
/// 0xac45a4010001a40200000000ffffffff
const N22_LO: u64 = 0x00000000ffffffff;
const N22_HI: u64 = 0xac45a4010001a402;

/// Precomputed constant g = round(N22 · 2^256 / r) for efficient Babai
/// rounding. Used to approximate c2 = round(k · N22 / r) as c2 ≈ (k · g) >>
/// 256.
const G_LIMBS: [u64; 3] =
    [0x63f6e522f6cfee2e, 0x7c6becf1e01faadd, 0x0000000000000001];

/// Fq modulus in 48-byte little-endian format (for point negation: y → q - y).
const Q_MODULUS_LE: [u8; 48] = [
    0xab, 0xaa, 0xff, 0xff, 0xff, 0xff, 0xfe, 0xb9, 0xff, 0xff, 0x53, 0xb1,
    0xfe, 0xff, 0xab, 0x1e, 0x24, 0xf6, 0xb0, 0xf6, 0xa0, 0xd2, 0x30, 0x67,
    0xbf, 0x12, 0x85, 0xf3, 0x84, 0x4b, 0x77, 0x64, 0xd7, 0xac, 0x4b, 0x43,
    0xb6, 0xa7, 0x1b, 0x4b, 0x9a, 0xe6, 0x7f, 0x39, 0xea, 0x11, 0x01, 0x1a,
];

/// Half of the scalar field modulus r, used to determine c1 = round(k/r).
/// r/2 = 0x39f6d3a994cebea241 9cec0404d0ec029defd2017fff2dff7fffffff80000000
const R_HALF: [u64; 4] = [
    0x7fffffff80000000,
    0x9defd2017fff2dff,
    0x9cec0404d0ec0402,
    0x39f6d3a994cebea2,
];

// ============================================================================
// GLV scalar decomposition
// ============================================================================

/// Decomposes a 256-bit scalar k into two ~128-bit scalars (k1, k2) such that
/// k ≡ k1 + k2·λ (mod r), where λ is the endomorphism eigenvalue.
///
/// Returns (|k1|, k1_is_negative, |k2|, k2_is_negative).
///
/// Note: The constants are BLS12-381-specific. This function works with any
/// `PrimeField` that has the same scalar field as BLS12-381.
pub fn glv_decompose<F: PrimeField>(k: &F) -> (u128, bool, u128, bool) {
    let k_repr = k.to_repr();
    let k_bytes = k_repr.as_ref();
    let k_limbs = bytes_to_u64x4(k_bytes);

    // c1 = round(k / r): since 0 ≤ k < r, c1 is 0 if k < r/2, else 1.
    let c1: u64 = if gt_u256(&k_limbs, &R_HALF) { 1 } else { 0 };

    // c2 = (k · g) >> 256 ≈ round(k · N22 / r)
    let c2 = mul_u256_u192_high(&k_limbs, &G_LIMBS);

    // k1 = k - c1 - c2 * N11  (signed, fits in ±2^128)
    let (k1_abs, k1_neg) = compute_k1(&k_limbs, c1, c2);

    // k2 = c1 * N22 - c2  (signed, fits in ±2^128)
    let (k2_abs, k2_neg) = compute_k2(c1, c2);

    (k1_abs, k1_neg, k2_abs, k2_neg)
}

/// Computes k1 = k - c1 - c2 * N11, returning (|k1|, is_negative).
fn compute_k1(k: &[u64; 4], c1: u64, c2: u128) -> (u128, bool) {
    // c2 * N11 as [u64; 4] (up to 256 bits)
    let prod = mul_u128_u128(c2, (N11_LO as u128) | ((N11_HI as u128) << 64));

    // k - c1 as [u64; 4] (with full borrow propagation)
    let (d0, b0) = k[0].overflowing_sub(c1);
    let (d1, b1) = k[1].overflowing_sub(b0 as u64);
    let (d2, b2) = k[2].overflowing_sub(b1 as u64);
    let d3 = k[3].wrapping_sub(b2 as u64);
    let k_minus_c1 = [d0, d1, d2, d3];

    // result = k_minus_c1 - prod (signed)
    let (diff, borrow) = sub_u256(&k_minus_c1, &prod);

    if borrow {
        // Result is negative: negate the 256-bit value
        let neg = negate_u256(&diff);
        let val = neg[0] as u128 | ((neg[1] as u128) << 64);
        debug_assert!(
            neg[2] == 0 && neg[3] == 0,
            "k1 overflow: doesn't fit in 128 bits"
        );
        (val, true)
    } else {
        let val = diff[0] as u128 | ((diff[1] as u128) << 64);
        debug_assert!(
            diff[2] == 0 && diff[3] == 0,
            "k1 overflow: doesn't fit in 128 bits"
        );
        (val, false)
    }
}

/// Computes k2 = c1 * N22 - c2, returning (|k2|, is_negative).
fn compute_k2(c1: u64, c2: u128) -> (u128, bool) {
    let n22 = (N22_LO as u128) | ((N22_HI as u128) << 64);
    let term = if c1 == 1 { n22 } else { 0u128 };

    if term >= c2 {
        (term - c2, false)
    } else {
        (c2 - term, true)
    }
}

// ============================================================================
// G1 endomorphism
// ============================================================================

/// Applies the GLV endomorphism φ(x, y) = (β·x, y) to a G1 affine point.
///
/// See also [`endomorphism_g1_bytes`] for the GPU-serialized-format equivalent
/// that avoids full deserialize/serialize overhead.
pub fn endomorphism_g1(p: &G1Affine) -> G1Affine {
    if bool::from(p.is_identity()) {
        return G1Affine::identity();
    }

    let beta: Fp = Fp::from_bytes_le(&BETA_LE_BYTES)
        .expect("BETA constant is a valid Fp element");

    // φ(x, y) = (β * x, y)
    let beta_x = beta * p.x();
    let y = p.y();

    // Reconstruct affine point via uncompressed serialization.
    // BLS12-381 uncompressed format: x_BE(48 bytes) || y_BE(48 bytes)
    let beta_x_be = beta_x.to_bytes_be();
    let y_be = y.to_bytes_be();

    let mut uncompressed = [0u8; 96];
    uncompressed[..48].copy_from_slice(&beta_x_be);
    uncompressed[48..].copy_from_slice(&y_be);

    // The point is guaranteed to be on the curve (endomorphism preserves this),
    // so unchecked is safe and avoids the expensive subgroup check.
    G1Affine::from_uncompressed_unchecked(&uncompressed)
        .expect("endomorphism produces a valid curve point")
}

// ============================================================================
// Serialized point utilities
// ============================================================================

/// Applies the GLV endomorphism φ(P) = (β·x, y) to a serialized G1 point.
///
/// This is the byte-level equivalent of [`endomorphism_g1`], operating directly
/// on the 30×13-bit limb GPU representation to avoid full
/// deserialize/serialize.
///
/// Format: x[120+8pad] || y[120+8pad] || z[120+8pad] = 384 bytes (each coord is
/// 30 LE u32s + 8 bytes padding from `@size(128)` in WGSL).
/// Returns new serialized bytes with x replaced by β·x.
pub fn endomorphism_g1_bytes(point_bytes: &[u8]) -> [u8; G1_GPU_BYTES] {
    debug_assert_eq!(point_bytes.len(), G1_GPU_BYTES);

    let mut result = [0u8; G1_GPU_BYTES];
    result.copy_from_slice(point_bytes);

    // Point at infinity (z = 0): check z data bytes (120 bytes at offset 2*128)
    let z_start = 2 * FQ_GPU_PADDED_BYTES;
    if point_bytes[z_start..z_start + FQ_GPU_BYTES]
        .iter()
        .all(|&b| b == 0)
    {
        return result;
    }

    // Convert x from 13-bit limb format (120 bytes at offset 0) to 48-byte LE
    let x_le = fq_13bit_to_bytes(&point_bytes[0..FQ_GPU_BYTES]);
    let x_le_arr: [u8; 48] = x_le.try_into().unwrap();
    let x = Fp::from_bytes_le(&x_le_arr).expect("valid Fp x-coordinate");

    let beta = Fp::from_bytes_le(&BETA_LE_BYTES).expect("valid BETA constant");
    let beta_x = beta * x;

    // Convert β·x back to 48-byte LE, then to 13-bit limb format
    let beta_x_be = beta_x.to_bytes_be();
    let mut beta_x_le = [0u8; 48];
    for i in 0..48 {
        beta_x_le[i] = beta_x_be[47 - i];
    }
    let beta_x_13bit = fq_bytes_to_13bit(&beta_x_le);
    result[0..FQ_GPU_BYTES].copy_from_slice(&beta_x_13bit);

    result
}

/// Negates a serialized G1 affine point in-place: (x, y, z) → (x, q−y, z).
///
/// The point is in the GPU 13-bit limb format with `@size(128)` padding:
/// x[120+8pad] || y[120+8pad] || z[120+8pad] = 384 bytes.
pub fn negate_g1_bytes(point_bytes: &mut [u8]) {
    debug_assert!(point_bytes.len() == G1_GPU_BYTES);

    // Check for point at infinity (all zeros)
    if point_bytes.iter().all(|&b| b == 0) {
        return;
    }

    // y is at bytes [FQ_GPU_PADDED_BYTES..FQ_GPU_PADDED_BYTES+FQ_GPU_BYTES] in
    // 13-bit limb format. Convert y from 13-bit (120 bytes) to 48-byte LE,
    // negate, convert back.
    let y_start = FQ_GPU_PADDED_BYTES;
    let y_le = fq_13bit_to_bytes(&point_bytes[y_start..y_start + FQ_GPU_BYTES]);

    // Compute q - y using 384-bit subtraction with borrow.
    let mut neg_y_le = [0u8; 48];
    let mut borrow: u16 = 0;
    for i in 0..48 {
        let q_byte = Q_MODULUS_LE[i] as u16;
        let y_byte = y_le[i] as u16;
        let diff = q_byte.wrapping_sub(y_byte).wrapping_sub(borrow);
        if q_byte < y_byte + borrow {
            neg_y_le[i] = diff as u8;
            borrow = 1;
        } else {
            neg_y_le[i] = diff as u8;
            borrow = 0;
        }
    }
    debug_assert_eq!(borrow, 0, "q - y underflow: y >= q");

    // Convert negated y back to 13-bit limb format and write in-place.
    let neg_y_13bit = fq_bytes_to_13bit(&neg_y_le);
    point_bytes[y_start..y_start + FQ_GPU_BYTES].copy_from_slice(&neg_y_13bit);
}

/// Decomposes a 128-bit unsigned integer into c-bit windows.
/// Returns `ceil(128/c)` windows, each in [0, 2^c).
pub fn u128_to_windows(k: u128, c: usize) -> Vec<u32> {
    let num_windows = 128_usize.div_ceil(c);
    let bytes = k.to_le_bytes();
    let mut windows = Vec::with_capacity(num_windows);
    let mask = (1u64 << c) - 1;

    for i in 0..num_windows {
        let bit_offset = i * c;
        let mut window: u64 = 0;
        let bit_shift = bit_offset % 8;
        let bytes_needed = (bit_shift + c).div_ceil(8).min(8);

        for j in 0..bytes_needed {
            let byte_idx = bit_offset / 8 + j;
            if byte_idx < 16 {
                window |= (bytes[byte_idx] as u64) << (j * 8);
            }
        }

        windows.push(((window >> bit_shift) & mask) as u32);
    }

    windows
}

/// Decomposes a 128-bit unsigned integer into signed c-bit windows.
/// Returns `ceil(128/c) + 1` entries (extra for carry), each as (|value|,
/// is_negative). Window values are in [0, 2^(c-1)], halving the bucket count vs
/// unsigned.
pub fn u128_to_signed_windows(k: u128, c: usize) -> Vec<(u32, bool)> {
    let mut unsigned = u128_to_windows(k, c);
    // Add extra window for potential carry from the last window
    unsigned.push(0);

    let half = 1u32 << (c - 1);
    let full = 1u32 << c;
    let mut result = Vec::with_capacity(unsigned.len());

    for i in 0..unsigned.len() {
        let w = unsigned[i];
        if w >= half {
            // Convert to negative digit: w - 2^c, carry +1 to next
            let abs_val = full - w;
            result.push((abs_val, true));
            if i + 1 < unsigned.len() {
                unsigned[i + 1] += 1;
            }
        } else {
            result.push((w, false));
        }
    }

    // Trim trailing zeros
    while result.len() > 1 && result.last() == Some(&(0, false)) {
        result.pop();
    }

    result
}

// ============================================================================
// Multi-precision arithmetic helpers
// ============================================================================

fn bytes_to_u64x4(bytes: &[u8]) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        let offset = i * 8;
        if offset + 8 <= bytes.len() {
            *limb = u64::from_le_bytes(
                bytes[offset..offset + 8].try_into().unwrap(),
            );
        } else {
            // Handle partial last limb
            let mut buf = [0u8; 8];
            let end = bytes.len().min(offset + 8);
            buf[..end - offset].copy_from_slice(&bytes[offset..end]);
            *limb = u64::from_le_bytes(buf);
        }
    }
    limbs
}

/// Returns true if a > b for 256-bit unsigned values.
fn gt_u256(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] > b[i] {
            return true;
        }
        if a[i] < b[i] {
            return false;
        }
    }
    false // equal
}

/// Computes the high 128 bits of (k * g) >> 256, where k is [u64; 4] and g is
/// [u64; 3]. This gives the approximate Babai rounding coefficient c2.
fn mul_u256_u192_high(k: &[u64; 4], g: &[u64; 3]) -> u128 {
    // Schoolbook multiplication: result[i+j] += k[i] * g[j]
    // We need result[4] and result[5] (the bits >> 256).
    // result has at most 7 limbs (indices 0..6).
    let mut result = [0u64; 7];

    for i in 0..4 {
        let mut carry = 0u64;
        for (j, &gj) in g.iter().enumerate() {
            let idx = i + j;
            let (lo, hi) = mul_u64(k[i], gj);
            let (sum1, c1) = result[idx].overflowing_add(lo);
            let (sum2, c2) = sum1.overflowing_add(carry);
            result[idx] = sum2;
            carry = hi + c1 as u64 + c2 as u64;
        }
        result[i + 3] = result[i + 3].wrapping_add(carry);
    }

    result[4] as u128 | ((result[5] as u128) << 64)
}

/// 64×64 → 128-bit multiply, returning (lo, hi).
#[inline(always)]
fn mul_u64(a: u64, b: u64) -> (u64, u64) {
    let r = a as u128 * b as u128;
    (r as u64, (r >> 64) as u64)
}

/// u128 × u128 → [u64; 4] (256-bit result).
fn mul_u128_u128(a: u128, b: u128) -> [u64; 4] {
    let a_lo = a as u64;
    let a_hi = (a >> 64) as u64;
    let b_lo = b as u64;
    let b_hi = (b >> 64) as u64;

    let (r0, c0) = mul_u64(a_lo, b_lo);
    let (r1a, c1a) = mul_u64(a_lo, b_hi);
    let (r1b, c1b) = mul_u64(a_hi, b_lo);
    let (r2, c2) = mul_u64(a_hi, b_hi);

    // Combine: result = r0 + (r1a + r1b) << 64 + (c0 + c1a + c1b + r2) << 128 +
    // c2 << 192
    let (mid_sum, mid_carry) = r1a.overflowing_add(r1b);
    let (r1, carry1) = mid_sum.overflowing_add(c0);

    let high_carry = mid_carry as u64 + carry1 as u64 + c1a + c1b;
    let (r2_sum, carry2) = r2.overflowing_add(high_carry);
    let r3 = c2 + carry2 as u64;

    [r0, r1, r2_sum, r3]
}

/// 256-bit subtraction: a - b, returning (result, borrow).
fn sub_u256(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
    let mut result = [0u64; 4];
    let mut borrow = 0u64;

    for i in 0..4 {
        let (diff1, b1) = a[i].overflowing_sub(b[i]);
        let (diff2, b2) = diff1.overflowing_sub(borrow);
        result[i] = diff2;
        borrow = b1 as u64 + b2 as u64;
    }

    (result, borrow != 0)
}

/// Negate a 256-bit value (two's complement).
fn negate_u256(a: &[u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    let mut carry = 1u64;
    for i in 0..4 {
        let (sum, c) = (!a[i]).overflowing_add(carry);
        result[i] = sum;
        carry = c as u64;
    }
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use blstrs::Scalar;
    use ff::Field;
    use group::Curve;

    use super::*;

    /// Verify that β is a valid cube root of unity in Fq.
    #[test]
    fn beta_is_cube_root_of_unity() {
        let beta: Fp = Fp::from_bytes_le(&BETA_LE_BYTES).unwrap();
        let one: Fp = Fp::ONE;
        let beta_cubed = beta * beta * beta;
        assert_eq!(beta_cubed, one, "β³ should equal 1");
        assert_ne!(beta, one, "β should not be 1");
    }

    /// Verify GLV decomposition: k1 + k2·λ ≡ k (mod r) for random scalars.
    #[test]
    fn glv_decompose_roundtrip() {
        // λ = -x² mod r (the endomorphism eigenvalue in Fr)
        let lambda = Scalar::from_repr_vartime([
            0x01, 0x00, 0x00, 0x00, 0xfe, 0xff, 0xff, 0xff, 0xfc, 0xb7, 0xfc,
            0xff, 0x01, 0x00, 0x78, 0xa7, 0x04, 0xd8, 0xa1, 0x09, 0x08, 0xd8,
            0x39, 0x33, 0x48, 0x7d, 0x9d, 0x29, 0x53, 0xa7, 0xed, 0x73,
        ])
        .expect("LAMBDA is a valid scalar");

        // Verify λ³ = 1
        let lambda_cubed = lambda * lambda * lambda;
        assert_eq!(lambda_cubed, Scalar::ONE, "λ³ should equal 1");

        let test_scalars = [
            Scalar::ZERO,
            Scalar::ONE,
            Scalar::from(2u64),
            Scalar::from(0xd201000000010000u64),
            -Scalar::ONE,
            lambda,
            Scalar::ROOT_OF_UNITY,
        ];

        for k in &test_scalars {
            let (k1_abs, k1_neg, k2_abs, k2_neg) = glv_decompose(k);

            // Reconstruct: k1 + k2 * lambda
            let k1_scalar = scalar_from_u128(k1_abs, k1_neg);
            let k2_scalar = scalar_from_u128(k2_abs, k2_neg);
            let reconstructed = k1_scalar + k2_scalar * lambda;
            assert_eq!(reconstructed, *k, "decomposition failed for k={:?}", k);
        }
    }

    /// Verify that decomposed components fit in 128 bits.
    #[test]
    fn glv_decompose_bounds() {
        use rand_core::OsRng;
        for _ in 0..1000 {
            let k = Scalar::random(&mut OsRng);
            let (k1, _, k2, _) = glv_decompose(&k);
            assert!(
                k1 < (1u128 << 127) + (1u128 << 126),
                "k1 too large: {} bits",
                128 - k1.leading_zeros()
            );
            assert!(
                k2 < (1u128 << 127) + (1u128 << 126),
                "k2 too large: {} bits",
                128 - k2.leading_zeros()
            );
        }
    }

    /// Verify the endomorphism property: φ(P) = [λ]·P for the generator.
    #[test]
    fn endomorphism_matches_scalar_mul() {
        let lambda = Scalar::from_repr_vartime([
            0x01, 0x00, 0x00, 0x00, 0xfe, 0xff, 0xff, 0xff, 0xfc, 0xb7, 0xfc,
            0xff, 0x01, 0x00, 0x78, 0xa7, 0x04, 0xd8, 0xa1, 0x09, 0x08, 0xd8,
            0x39, 0x33, 0x48, 0x7d, 0x9d, 0x29, 0x53, 0xa7, 0xed, 0x73,
        ])
        .unwrap();

        let g = G1Affine::generator();
        let phi_g = endomorphism_g1(&g);
        let lambda_g: G1Affine =
            (blstrs::G1Projective::from(g) * lambda).to_affine();

        assert_eq!(phi_g, lambda_g, "φ(G) should equal [λ]·G");
    }

    /// Verify the endomorphism on the identity point.
    #[test]
    fn endomorphism_identity() {
        let inf = G1Affine::identity();
        assert_eq!(endomorphism_g1(&inf), inf);
    }

    /// Verify that endomorphism_g1_bytes produces the same result as
    /// endomorphism_g1.
    #[test]
    fn endomorphism_g1_bytes_matches_affine() {
        use crate::gpu::curve::GpuCurve;

        let g = G1Affine::generator();
        let phi_g = endomorphism_g1(&g);

        // Apply endomorphism via byte-level function
        let g_bytes = <blstrs::Bls12 as GpuCurve>::serialize_g1(&g);
        let phi_bytes = endomorphism_g1_bytes(&g_bytes);

        // Deserialize and compare
        let phi_from_bytes =
            <blstrs::Bls12 as GpuCurve>::deserialize_g1(&phi_bytes)
                .expect("endomorphism bytes should deserialize");
        let phi_affine = blstrs::G1Projective::from(phi_g);

        assert_eq!(
            phi_from_bytes, phi_affine,
            "byte-level endomorphism should match affine-level"
        );
    }

    /// Verify that negating serialized bytes twice returns the original.
    #[test]
    fn negate_g1_bytes_roundtrip() {
        use crate::gpu::curve::GpuCurve;
        let p = G1Affine::generator();
        let mut bytes = <blstrs::Bls12 as GpuCurve>::serialize_g1(&p);
        let original = bytes.clone();

        negate_g1_bytes(&mut bytes);
        assert_ne!(bytes, original, "negation should change the point");

        negate_g1_bytes(&mut bytes);
        assert_eq!(bytes, original, "double negation should return original");
    }

    /// Verify that negated bytes deserialize to the group-negated point.
    #[test]
    fn negate_g1_bytes_matches_group_negation() {
        use crate::gpu::curve::GpuCurve;
        let p = G1Affine::generator();
        let mut bytes = <blstrs::Bls12 as GpuCurve>::serialize_g1(&p);
        negate_g1_bytes(&mut bytes);

        let deserialized = <blstrs::Bls12 as GpuCurve>::deserialize_g1(&bytes)
            .expect("negated point should deserialize");
        let neg_p: blstrs::G1Projective =
            (-blstrs::G1Projective::from(p)).into();

        assert_eq!(deserialized, neg_p, "negated bytes should match -P");
    }

    /// Test u128_to_windows with known values.
    #[test]
    fn u128_to_windows_known_values() {
        // c=15: ceil(128/15) = 9 windows
        let windows = u128_to_windows(0, 15);
        assert_eq!(windows.len(), 9);
        assert!(windows.iter().all(|&w| w == 0));

        let windows = u128_to_windows(1, 15);
        assert_eq!(windows[0], 1);
        assert!(windows[1..].iter().all(|&w| w == 0));

        // 2^15 - 1 = 32767 in window 0
        let windows = u128_to_windows(32767, 15);
        assert_eq!(windows[0], 32767);
        assert!(windows[1..].iter().all(|&w| w == 0));

        // 2^15 = 1 in window 1
        let windows = u128_to_windows(1 << 15, 15);
        assert_eq!(windows[0], 0);
        assert_eq!(windows[1], 1);
    }

    /// Test u128_to_windows matches bit-by-bit extraction.
    #[test]
    fn u128_to_windows_matches_bit_extraction() {
        let vals = [
            0u128,
            1,
            0xdeadbeef,
            u128::MAX,
            1u128 << 127,
            (1u128 << 127) - 1,
        ];
        for val in vals {
            let windows = u128_to_windows(val, 15);
            let expected = reference_u128_windows(val, 15);
            assert_eq!(windows, expected, "mismatch for val={}", val);
        }
    }

    fn reference_u128_windows(k: u128, c: usize) -> Vec<u32> {
        let num_windows = 128_usize.div_ceil(c);
        let mut out = Vec::with_capacity(num_windows);
        for i in 0..num_windows {
            let bit_offset = i * c;
            let mut w = 0u32;
            for j in 0..c {
                let idx = bit_offset + j;
                if idx < 128 && (k >> idx) & 1 == 1 {
                    w |= 1u32 << j;
                }
            }
            out.push(w);
        }
        out
    }

    /// Verify β² + β + 1 = 0 (characteristic polynomial of cube root of unity).
    #[test]
    fn beta_satisfies_minimal_polynomial() {
        let beta: Fp = Fp::from_bytes_le(&BETA_LE_BYTES).unwrap();
        // β² + β + 1 should be 0 in Fq
        let beta_sq = beta * beta;
        let sum = beta_sq + beta + Fp::ONE;
        assert_eq!(sum, Fp::ZERO, "β should satisfy β² + β + 1 = 0");
    }

    /// Verify GLV decomposition for k=1 (simplest non-trivial case).
    #[test]
    fn glv_decompose_one() {
        let lambda = Scalar::from_repr_vartime([
            0x01, 0x00, 0x00, 0x00, 0xfe, 0xff, 0xff, 0xff, 0xfc, 0xb7, 0xfc,
            0xff, 0x01, 0x00, 0x78, 0xa7, 0x04, 0xd8, 0xa1, 0x09, 0x08, 0xd8,
            0x39, 0x33, 0x48, 0x7d, 0x9d, 0x29, 0x53, 0xa7, 0xed, 0x73,
        ])
        .unwrap();

        let (k1, k1_neg, k2, k2_neg) = glv_decompose(&Scalar::ONE);
        let k1_s = scalar_from_u128(k1, k1_neg);
        let k2_s = scalar_from_u128(k2, k2_neg);
        let reconstructed = k1_s + k2_s * lambda;
        assert_eq!(reconstructed, Scalar::ONE, "decomposition of 1 failed");
    }

    /// Verify GLV decomposition for r-1 (largest scalar).
    #[test]
    fn glv_decompose_r_minus_one() {
        let lambda = Scalar::from_repr_vartime([
            0x01, 0x00, 0x00, 0x00, 0xfe, 0xff, 0xff, 0xff, 0xfc, 0xb7, 0xfc,
            0xff, 0x01, 0x00, 0x78, 0xa7, 0x04, 0xd8, 0xa1, 0x09, 0x08, 0xd8,
            0x39, 0x33, 0x48, 0x7d, 0x9d, 0x29, 0x53, 0xa7, 0xed, 0x73,
        ])
        .unwrap();

        let k = -Scalar::ONE; // r - 1
        let (k1, k1_neg, k2, k2_neg) = glv_decompose(&k);

        // Verify reconstruction
        let k1_s = scalar_from_u128(k1, k1_neg);
        let k2_s = scalar_from_u128(k2, k2_neg);
        let reconstructed = k1_s + k2_s * lambda;
        assert_eq!(reconstructed, k, "decomposition of r-1 failed");

        // Both halves should fit in ~128 bits
        assert!(k1 < (1u128 << 127) + (1u128 << 126), "k1 too large for r-1");
        assert!(k2 < (1u128 << 127) + (1u128 << 126), "k2 too large for r-1");
    }

    /// Verify GLV decomposition for λ itself: k=λ should give k1=0, k2=1.
    #[test]
    fn glv_decompose_lambda() {
        let lambda = Scalar::from_repr_vartime([
            0x01, 0x00, 0x00, 0x00, 0xfe, 0xff, 0xff, 0xff, 0xfc, 0xb7, 0xfc,
            0xff, 0x01, 0x00, 0x78, 0xa7, 0x04, 0xd8, 0xa1, 0x09, 0x08, 0xd8,
            0x39, 0x33, 0x48, 0x7d, 0x9d, 0x29, 0x53, 0xa7, 0xed, 0x73,
        ])
        .unwrap();

        let (k1, k1_neg, k2, k2_neg) = glv_decompose(&lambda);
        let k1_s = scalar_from_u128(k1, k1_neg);
        let k2_s = scalar_from_u128(k2, k2_neg);
        let reconstructed = k1_s + k2_s * lambda;
        assert_eq!(reconstructed, lambda, "decomposition of λ failed");
    }

    /// Verify multi-precision arithmetic: mul_u128_u128.
    #[test]
    fn mul_u128_u128_known_values() {
        // 0 * 0 = 0
        let r = mul_u128_u128(0, 0);
        assert_eq!(r, [0, 0, 0, 0]);

        // 1 * 1 = 1
        let r = mul_u128_u128(1, 1);
        assert_eq!(r, [1, 0, 0, 0]);

        // (2^64) * (2^64) = 2^128
        let a: u128 = 1u128 << 64;
        let b: u128 = 1u128 << 64;
        let r = mul_u128_u128(a, b);
        // 2^128 = [0, 0, 1, 0]
        assert_eq!(r, [0, 0, 1, 0]);

        // (2^128 - 1) * 2 = 2^129 - 2
        let max = u128::MAX;
        let r = mul_u128_u128(max, 2);
        // max * 2 = 2^129 - 2 = [0xFFFF...FFFE, 0xFFFF...FFFF, 1, 0]
        assert_eq!(r[0], u64::MAX - 1); // 0xFFFFFFFFFFFFFFFE
        assert_eq!(r[1], u64::MAX);
        assert_eq!(r[2], 1);
        assert_eq!(r[3], 0);
    }

    /// Verify multi-precision arithmetic: sub_u256.
    #[test]
    fn sub_u256_known_values() {
        // a - 0 = a
        let a = [42u64, 0, 0, 0];
        let (r, borrow) = sub_u256(&a, &[0, 0, 0, 0]);
        assert_eq!(r, a);
        assert!(!borrow);

        // 0 - 1 borrows (underflow)
        let (r, borrow) = sub_u256(&[0, 0, 0, 0], &[1, 0, 0, 0]);
        assert!(borrow);
        assert_eq!(r[0], u64::MAX); // wraps around

        // a - a = 0
        let a = [0x1234, 0x5678, 0x9abc, 0xdef0];
        let (r, borrow) = sub_u256(&a, &a);
        assert_eq!(r, [0, 0, 0, 0]);
        assert!(!borrow);
    }

    /// Verify multi-precision arithmetic: negate_u256.
    #[test]
    fn negate_u256_known_values() {
        // -0 = 0
        let r = negate_u256(&[0, 0, 0, 0]);
        assert_eq!(r, [0, 0, 0, 0]);

        // -1 = MAX (two's complement)
        let r = negate_u256(&[1, 0, 0, 0]);
        assert_eq!(r, [u64::MAX, u64::MAX, u64::MAX, u64::MAX]);

        // double negation: -(-x) = x
        let x = [0xdeadbeef, 0xcafebabe, 0x12345678, 0x9abcdef0];
        let neg = negate_u256(&x);
        let double_neg = negate_u256(&neg);
        assert_eq!(double_neg, x);
    }

    /// Verify gt_u256 comparison.
    #[test]
    fn gt_u256_comparison() {
        // Equal values: not greater than
        let a = [1, 2, 3, 4];
        assert!(!gt_u256(&a, &a));

        // Differ in highest limb
        assert!(gt_u256(&[0, 0, 0, 5], &[0, 0, 0, 4]));
        assert!(!gt_u256(&[0, 0, 0, 4], &[0, 0, 0, 5]));

        // Differ in lowest limb only
        assert!(gt_u256(&[2, 0, 0, 0], &[1, 0, 0, 0]));
        assert!(!gt_u256(&[1, 0, 0, 0], &[2, 0, 0, 0]));

        // Higher limbs dominate
        assert!(gt_u256(&[0, 0, 0, 1], &[u64::MAX, u64::MAX, u64::MAX, 0]));
    }

    /// Verify bytes_to_u64x4 with known input.
    #[test]
    fn bytes_to_u64x4_roundtrip() {
        let limbs = [
            0x0102030405060708u64,
            0x090a0b0c0d0e0f10,
            0x1112131415161718,
            0x191a1b1c1d1e1f20,
        ];
        let mut bytes = Vec::new();
        for l in &limbs {
            bytes.extend_from_slice(&l.to_le_bytes());
        }
        let parsed = bytes_to_u64x4(&bytes);
        assert_eq!(parsed, limbs);
    }

    /// Verify u128_to_windows for various window sizes.
    #[test]
    fn u128_to_windows_various_widths() {
        for c in [8, 13, 15, 16] {
            let vals = [0u128, 1, u128::MAX, 1u128 << 127, (1u128 << 64) - 1];
            for val in vals {
                let windows = u128_to_windows(val, c);
                let expected = reference_u128_windows(val, c);
                assert_eq!(
                    windows, expected,
                    "mismatch for val={val:#x} c={c}"
                );
            }
        }
    }

    /// Endomorphism applied twice is not identity but applying three times is.
    #[test]
    fn endomorphism_g1_order_three() {
        let g = G1Affine::generator();
        let phi1 = endomorphism_g1(&g);
        let phi2 = endomorphism_g1(&phi1);
        let phi3 = endomorphism_g1(&phi2);

        assert_ne!(phi1, g, "φ(G) should not equal G");
        assert_ne!(phi2, g, "φ²(G) should not equal G");
        assert_eq!(phi3, g, "φ³(G) should equal G (order 3)");
    }

    /// Point negation on identity is a no-op.
    #[test]
    fn negate_g1_bytes_identity() {
        let mut zero_bytes = vec![0u8; G1_GPU_BYTES];
        let original = zero_bytes.clone();
        negate_g1_bytes(&mut zero_bytes);
        assert_eq!(zero_bytes, original, "negating identity should be a no-op");
    }

    /// Endomorphism and negation commute: φ(-P) = -φ(P).
    #[test]
    fn endomorphism_negation_commute() {
        use crate::gpu::curve::GpuCurve;

        let g = G1Affine::generator();
        let g_bytes = <blstrs::Bls12 as GpuCurve>::serialize_g1(&g);

        // φ(-P)
        let mut neg_bytes = g_bytes.clone();
        negate_g1_bytes(&mut neg_bytes);
        let phi_neg = endomorphism_g1_bytes(&neg_bytes);

        // -φ(P)
        let phi_bytes = endomorphism_g1_bytes(&g_bytes);
        let mut neg_phi = phi_bytes.clone();
        negate_g1_bytes(&mut neg_phi);

        // They should be equal (endomorphism is a group homomorphism)
        assert_eq!(
            phi_neg.to_vec(),
            neg_phi.to_vec(),
            "φ(-P) should equal -φ(P)"
        );
    }

    /// GLV decomposition for k=0.
    #[test]
    fn glv_decompose_zero() {
        let (k1, k1_neg, k2, k2_neg) = glv_decompose(&Scalar::ZERO);
        assert_eq!(k1, 0);
        assert_eq!(k2, 0);
        assert!(!k1_neg);
        assert!(!k2_neg);
    }

    fn scalar_from_u128(val: u128, negative: bool) -> Scalar {
        let lo = val as u64;
        let hi = (val >> 64) as u64;
        let mut bytes = [0u8; 32];
        bytes[0..8].copy_from_slice(&lo.to_le_bytes());
        bytes[8..16].copy_from_slice(&hi.to_le_bytes());
        let s = Scalar::from_repr_vartime(bytes)
            .expect("128-bit value is a valid scalar");
        if negative { -s } else { s }
    }
}
