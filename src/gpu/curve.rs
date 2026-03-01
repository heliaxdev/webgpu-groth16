use std::ops::{Add, Mul, Sub};

use ff::{Field, PrimeField, PrimeFieldBits};
use group::Group;
use group::prime::PrimeCurveAffine;

/// Size of a single Fq element in GPU format: 30 × 4 bytes = 120 bytes.
pub const FQ_GPU_BYTES: usize = 120;
/// Size of a G1 point in GPU format: 3 × 120 = 360 bytes.
pub const G1_GPU_BYTES: usize = 3 * FQ_GPU_BYTES;
/// Size of a G2 point in GPU format: 6 × 120 = 720 bytes.
pub const G2_GPU_BYTES: usize = 6 * FQ_GPU_BYTES;

/// Convert a 48-byte little-endian field element to 30×13-bit limb representation (120 bytes).
/// Each 13-bit limb is stored as a 4-byte little-endian u32.
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

/// Convert 30×13-bit limb representation (120 bytes) back to 48-byte little-endian.
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

pub trait GpuCurve: 'static {
    type Engine: pairing::Engine;

    type Scalar: PrimeField + PrimeFieldBits;
    type G1: Group<Scalar = Self::Scalar>;
    type G2: Group<Scalar = Self::Scalar>;
    type G1Affine;
    type G2Affine;

    const NTT_SOURCE: &'static str;
    const MSM_SOURCE: &'static str;
    const POLY_OPS_SOURCE: &'static str;

    // Serialization
    fn serialize_g1(point: &Self::G1Affine) -> Vec<u8>;
    fn serialize_g2(point: &Self::G2Affine) -> Vec<u8>;
    fn serialize_scalar(s: &Self::Scalar) -> Vec<u8>;
    fn deserialize_scalar(bytes: &[u8]) -> anyhow::Result<Self::Scalar>;
    fn deserialize_g1(bytes: &[u8]) -> anyhow::Result<Self::G1>;
    fn deserialize_g2(bytes: &[u8]) -> anyhow::Result<Self::G2>;

    // Scalar decomposition for bucket sorting
    fn scalar_to_windows(s: &Self::Scalar, c: usize) -> Vec<u32>;

    /// Signed-digit scalar decomposition: each window value is in `[-(2^(c-1)), 2^(c-1)]`.
    /// Returns `(absolute_value, is_negative)` pairs.  Bucket values are halved compared
    /// to unsigned windows, and points with negative windows are negated on the GPU.
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

    // Optimal bucket width for MSM (c such that 2^c buckets)
    fn bucket_width() -> usize;

    // Smaller bucket width for G2 MSM (avoids O(2^c) subsum without gap-skipping)
    fn g2_bucket_width() -> usize {
        8
    }

    // NTT support
    fn root_of_unity(n: usize) -> Self::Scalar;

    // Identity element
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

    fn serialize_g1(point: &Self::G1Affine) -> Vec<u8> {
        let is_inf: bool = point.is_identity().into();

        if is_inf {
            return vec![0u8; G1_GPU_BYTES];
        }

        let mut uncompressed = point.to_uncompressed();
        // Strip Zcash metadata bits from x
        uncompressed[0] &= 0b0001_1111;

        // Reverse Big-Endian 48-byte chunks to Little-Endian
        let mut x_le = uncompressed[0..48].to_vec();
        x_le.reverse();
        let mut y_le = uncompressed[48..96].to_vec();
        y_le.reverse();

        // z = 1 in standard form (not Montgomery)
        let mut z_le = vec![0u8; 48];
        z_le[0] = 1;

        // Convert each 48-byte LE coordinate to 120-byte 13-bit representation
        let mut wgsl_bytes = Vec::with_capacity(G1_GPU_BYTES);
        wgsl_bytes.extend_from_slice(&fq_bytes_to_13bit(&x_le));
        wgsl_bytes.extend_from_slice(&fq_bytes_to_13bit(&y_le));
        wgsl_bytes.extend_from_slice(&fq_bytes_to_13bit(&z_le));
        wgsl_bytes
    }

    fn serialize_g2(point: &Self::G2Affine) -> Vec<u8> {
        let is_inf: bool = point.is_identity().into();

        if is_inf {
            return vec![0u8; G2_GPU_BYTES];
        }

        let mut uncompressed = point.to_uncompressed();
        // Strip Zcash metadata bits from x.c1
        uncompressed[0] &= 0b0001_1111;

        // blstrs G2 uncompressed layout (192 bytes):
        // x.c1 (48 bytes, BE), x.c0 (48 bytes, BE)
        // y.c1 (48 bytes, BE), y.c0 (48 bytes, BE)
        let mut x_c1 = uncompressed[0..48].to_vec();
        x_c1.reverse();
        let mut x_c0 = uncompressed[48..96].to_vec();
        x_c0.reverse();
        let mut y_c1 = uncompressed[96..144].to_vec();
        y_c1.reverse();
        let mut y_c0 = uncompressed[144..192].to_vec();
        y_c0.reverse();

        // z = (1, 0) in standard form
        let mut z_c0 = vec![0u8; 48];
        z_c0[0] = 1;
        let z_c1 = vec![0u8; 48];

        // Convert each 48-byte LE coordinate to 120-byte 13-bit representation
        // Layout: x.c0, x.c1, y.c0, y.c1, z.c0, z.c1
        let mut wgsl_bytes = Vec::with_capacity(G2_GPU_BYTES);
        wgsl_bytes.extend_from_slice(&fq_bytes_to_13bit(&x_c0));
        wgsl_bytes.extend_from_slice(&fq_bytes_to_13bit(&x_c1));
        wgsl_bytes.extend_from_slice(&fq_bytes_to_13bit(&y_c0));
        wgsl_bytes.extend_from_slice(&fq_bytes_to_13bit(&y_c1));
        wgsl_bytes.extend_from_slice(&fq_bytes_to_13bit(&z_c0));
        wgsl_bytes.extend_from_slice(&fq_bytes_to_13bit(&z_c1));
        wgsl_bytes
    }

    fn deserialize_g1(bytes: &[u8]) -> anyhow::Result<Self::G1> {
        if bytes.len() != G1_GPU_BYTES {
            anyhow::bail!("Invalid G1 byte length from GPU: {}", bytes.len());
        }
        // GPU outputs 360 bytes: X (120), Y (120), Z (120) in 30×13-bit format.

        // 1. Check if the point is at infinity by inspecting the Z coordinate
        let z_bytes = &bytes[2 * FQ_GPU_BYTES..3 * FQ_GPU_BYTES];
        let mut z_is_zero = true;
        for &b in z_bytes {
            if b != 0 {
                z_is_zero = false;
                break;
            }
        }
        if z_is_zero {
            return Ok(Self::G1::identity());
        }

        // 2. Convert 13-bit limb representation back to 48-byte LE
        let x_le = fq_13bit_to_bytes(&bytes[0..FQ_GPU_BYTES]);
        let y_le = fq_13bit_to_bytes(&bytes[FQ_GPU_BYTES..2 * FQ_GPU_BYTES]);

        // 3. Reverse to Big-Endian for Zcash uncompressed format
        let mut x_be = x_le;
        x_be.reverse();
        let mut y_be = y_le;
        y_be.reverse();

        let mut uncompressed = [0u8; 96];
        uncompressed[..48].copy_from_slice(&x_be);
        uncompressed[48..].copy_from_slice(&y_be);

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

    fn deserialize_g2(bytes: &[u8]) -> anyhow::Result<Self::G2> {
        if bytes.len() != G2_GPU_BYTES {
            anyhow::bail!("Invalid G2 byte length from GPU: {}", bytes.len());
        }
        // GPU outputs 720 bytes: X_c0, X_c1, Y_c0, Y_c1, Z_c0, Z_c1 (all 120 bytes, 13-bit)

        // 1. Check if the point is at infinity by inspecting Z (last 2 × 120 bytes)
        let z_bytes = &bytes[4 * FQ_GPU_BYTES..6 * FQ_GPU_BYTES];
        let mut z_is_zero = true;
        for &b in z_bytes {
            if b != 0 {
                z_is_zero = false;
                break;
            }
        }
        if z_is_zero {
            return Ok(Self::G2::identity());
        }

        // 2. Convert each 120-byte 13-bit coordinate back to 48-byte LE
        let x_c0_le = fq_13bit_to_bytes(&bytes[0..FQ_GPU_BYTES]);
        let x_c1_le = fq_13bit_to_bytes(&bytes[FQ_GPU_BYTES..2 * FQ_GPU_BYTES]);
        let y_c0_le = fq_13bit_to_bytes(&bytes[2 * FQ_GPU_BYTES..3 * FQ_GPU_BYTES]);
        let y_c1_le = fq_13bit_to_bytes(&bytes[3 * FQ_GPU_BYTES..4 * FQ_GPU_BYTES]);

        // 3. Reverse to Big-Endian for Zcash uncompressed format
        let mut x_c0_be = x_c0_le;
        x_c0_be.reverse();
        let mut x_c1_be = x_c1_le;
        x_c1_be.reverse();
        let mut y_c0_be = y_c0_le;
        y_c0_be.reverse();
        let mut y_c1_be = y_c1_le;
        y_c1_be.reverse();

        // Reconstruct Zcash BE uncompressed structure: x_c1, x_c0, y_c1, y_c0
        let mut uncompressed = [0u8; 192];
        uncompressed[0..48].copy_from_slice(&x_c1_be);
        uncompressed[48..96].copy_from_slice(&x_c0_be);
        uncompressed[96..144].copy_from_slice(&y_c1_be);
        uncompressed[144..192].copy_from_slice(&y_c0_be);

        let ct: subtle::CtOption<blstrs::G2Affine> =
            blstrs::G2Affine::from_uncompressed(&uncompressed);
        let affine: Option<blstrs::G2Affine> = ct.into();

        if let Some(affine) = affine {
            Ok(affine.into())
        } else {
            anyhow::bail!("Failed to deserialize G2 point from GPU")
        }
    }

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
