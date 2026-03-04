//! BLS12-381-specific GPU curve implementation.

use std::ops::{Add, Mul, Sub};

use ff::{Field, PrimeField, PrimeFieldBits};
use group::Group;
use group::prime::PrimeCurveAffine;

use super::{G1MsmDecomposition, GlvWindowDecomposition, GpuCurve};

const FQ_COORD_SIZE: usize = 48;
const ZCASH_METADATA_MASK: u8 = 0b0001_1111;
const BLS12_FQ_GPU_BYTES: usize = 120;
const BLS12_FQ_GPU_PADDED_BYTES: usize = 128;
const BLS12_G1_GPU_BYTES: usize = 3 * BLS12_FQ_GPU_PADDED_BYTES;
const BLS12_G2_GPU_BYTES: usize = 6 * BLS12_FQ_GPU_BYTES;

pub fn fq_bytes_to_13bit(bytes_48: &[u8]) -> Vec<u8> {
    debug_assert_eq!(bytes_48.len(), 48);
    let mut result = vec![0u8; BLS12_FQ_GPU_BYTES];
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

pub fn fq_13bit_to_bytes(bytes_120: &[u8]) -> Vec<u8> {
    debug_assert_eq!(bytes_120.len(), BLS12_FQ_GPU_BYTES);
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

fn scalar_to_le_bytes(s: &blstrs::Scalar) -> [u8; 32] {
    let bits = s.to_le_bits();
    let mut bytes = [0u8; 32];
    for (i, bit) in bits.iter().enumerate() {
        if *bit {
            bytes[i / 8] |= 1 << (i % 8);
        }
    }
    bytes
}

fn be_coord_to_gpu_limbs(be_bytes: &[u8]) -> Vec<u8> {
    let mut le = be_bytes.to_vec();
    le.reverse();
    fq_bytes_to_13bit(&le)
}

fn gpu_limbs_to_be_coord(limb_bytes: &[u8]) -> Vec<u8> {
    let mut be = fq_13bit_to_bytes(limb_bytes);
    be.reverse();
    be
}

fn is_all_zero(bytes: &[u8]) -> bool {
    bytes.iter().all(|&b| b == 0)
}

impl GpuCurve for blstrs::Bls12 {
    type Engine = Self;
    type G1 = <Self::Engine as pairing::Engine>::G1;
    type G1Affine = <Self::Engine as pairing::Engine>::G1Affine;
    type G2 = <Self::Engine as pairing::Engine>::G2;
    type G2Affine = <Self::Engine as pairing::Engine>::G2Affine;
    type Scalar = <Self::Engine as pairing::Engine>::Fr;

    const FQ_GPU_BYTES: usize = BLS12_FQ_GPU_BYTES;
    const FQ_GPU_PADDED_BYTES: usize = BLS12_FQ_GPU_PADDED_BYTES;
    const G1_GPU_BYTES: usize = BLS12_G1_GPU_BYTES;
    const G1_SUBSUM_CHUNKS_PER_WINDOW: u32 = 1;
    const G2_GPU_BYTES: usize = BLS12_G2_GPU_BYTES;
    const G2_SUBSUM_CHUNKS_PER_WINDOW: u32 = 32;
    const HAS_G1_GLV: bool = true;
    const MSM_G1_AGG_SOURCE: &'static str = concat!(
        include_str!("../../shader/bls12_381/fr.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/fp.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/curve_g1.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/msm_g1_agg.wgsl"),
    );
    const MSM_G1_SUBSUM_SOURCE: &'static str = concat!(
        include_str!("../../shader/bls12_381/fr.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/fp.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/curve_g1.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/msm_g1_subsum.wgsl"),
    );
    const MSM_G2_AGG_SOURCE: &'static str = concat!(
        include_str!("../../shader/bls12_381/fr.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/fp.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/curve_g2.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/msm_g2_agg.wgsl"),
    );
    const MSM_G2_SUBSUM_SOURCE: &'static str = concat!(
        include_str!("../../shader/bls12_381/fr.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/fp.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/curve_g2.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/msm_g2_subsum.wgsl"),
    );
    const MSM_INDEX_SIGN_BIT: u32 = 1 << 31;
    const MSM_MAX_CHUNK_SIZE: u32 = 64;
    const MSM_WORKGROUP_SIZE: u32 = 64;
    const NTT_FUSED_SOURCE: &'static str = concat!(
        include_str!("../../shader/bls12_381/fr.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/fp.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/ntt.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/ntt_fused.wgsl"),
    );
    const NTT_SOURCE: &'static str = concat!(
        include_str!("../../shader/bls12_381/fr.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/fp.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/ntt.wgsl"),
    );
    const NTT_TILE_SIZE: u32 = 512;
    const POLY_OPS_SOURCE: &'static str = concat!(
        include_str!("../../shader/bls12_381/fr.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/fp.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/poly_ops.wgsl"),
    );
    const SCALAR_WORKGROUP_SIZE: u32 = 256;
    #[cfg(test)]
    const TEST_SHADER_G1_SOURCE: &'static str = concat!(
        include_str!("../../shader/bls12_381/fr.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/fp.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/curve_g1.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/msm_g1_subsum.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/msm_test_debug_g1.wgsl"),
    );
    #[cfg(test)]
    const TEST_SHADER_G2_SOURCE: &'static str = concat!(
        include_str!("../../shader/bls12_381/fr.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/fp.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/curve_g2.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/msm_g2_subsum.wgsl"),
        "\n",
        include_str!("../../shader/bls12_381/msm_test_debug_g2.wgsl"),
    );

    fn serialize_g1(point: &Self::G1Affine) -> Vec<u8> {
        let is_inf: bool = point.is_identity().into();
        if is_inf {
            return vec![0u8; Self::G1_GPU_BYTES];
        }

        let mut uncompressed = point.to_uncompressed();
        uncompressed[0] &= ZCASH_METADATA_MASK;

        let mut z_le = vec![0u8; FQ_COORD_SIZE];
        z_le[0] = 1;

        let mut wgsl_bytes = Vec::with_capacity(Self::G1_GPU_BYTES);
        wgsl_bytes.extend_from_slice(&be_coord_to_gpu_limbs(
            &uncompressed[..FQ_COORD_SIZE],
        ));
        wgsl_bytes.extend_from_slice(&[0u8; 8]);
        wgsl_bytes.extend_from_slice(&be_coord_to_gpu_limbs(
            &uncompressed[FQ_COORD_SIZE..2 * FQ_COORD_SIZE],
        ));
        wgsl_bytes.extend_from_slice(&[0u8; 8]);
        wgsl_bytes.extend_from_slice(&fq_bytes_to_13bit(&z_le));
        wgsl_bytes.extend_from_slice(&[0u8; 8]);
        wgsl_bytes
    }

    fn serialize_g2(point: &Self::G2Affine) -> Vec<u8> {
        let is_inf: bool = point.is_identity().into();
        if is_inf {
            return vec![0u8; Self::G2_GPU_BYTES];
        }

        let mut uncompressed = point.to_uncompressed();
        uncompressed[0] &= ZCASH_METADATA_MASK;

        let s = FQ_COORD_SIZE;
        let x_c1 = be_coord_to_gpu_limbs(&uncompressed[0..s]);
        let x_c0 = be_coord_to_gpu_limbs(&uncompressed[s..2 * s]);
        let y_c1 = be_coord_to_gpu_limbs(&uncompressed[2 * s..3 * s]);
        let y_c0 = be_coord_to_gpu_limbs(&uncompressed[3 * s..4 * s]);

        let mut z_c0_le = vec![0u8; FQ_COORD_SIZE];
        z_c0_le[0] = 1;
        let z_c1_le = vec![0u8; FQ_COORD_SIZE];

        let mut wgsl_bytes = Vec::with_capacity(Self::G2_GPU_BYTES);
        wgsl_bytes.extend_from_slice(&x_c0);
        wgsl_bytes.extend_from_slice(&x_c1);
        wgsl_bytes.extend_from_slice(&y_c0);
        wgsl_bytes.extend_from_slice(&y_c1);
        wgsl_bytes.extend_from_slice(&fq_bytes_to_13bit(&z_c0_le));
        wgsl_bytes.extend_from_slice(&fq_bytes_to_13bit(&z_c1_le));
        wgsl_bytes
    }

    fn deserialize_g1(bytes: &[u8]) -> anyhow::Result<Self::G1> {
        if bytes.len() != Self::G1_GPU_BYTES {
            anyhow::bail!("Invalid G1 byte length from GPU: {}", bytes.len());
        }
        if is_all_zero(
            &bytes[2 * Self::FQ_GPU_PADDED_BYTES
                ..2 * Self::FQ_GPU_PADDED_BYTES + Self::FQ_GPU_BYTES],
        ) {
            return Ok(Self::G1::identity());
        }

        let x_be = gpu_limbs_to_be_coord(&bytes[0..Self::FQ_GPU_BYTES]);
        let y_be = gpu_limbs_to_be_coord(
            &bytes[Self::FQ_GPU_PADDED_BYTES
                ..Self::FQ_GPU_PADDED_BYTES + Self::FQ_GPU_BYTES],
        );

        let mut uncompressed = [0u8; 96];
        uncompressed[..FQ_COORD_SIZE].copy_from_slice(&x_be);
        uncompressed[FQ_COORD_SIZE..].copy_from_slice(&y_be);

        let ct: subtle::CtOption<blstrs::G1Affine> =
            blstrs::G1Affine::from_uncompressed(&uncompressed);
        let affine: Option<blstrs::G1Affine> = ct.into();

        if let Some(affine) = affine {
            Ok(affine.into())
        } else {
            let limb0 =
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            anyhow::bail!(
                "Failed to deserialize G1 point from GPU (x_limb0={:#x})",
                limb0
            )
        }
    }

    fn deserialize_g2(bytes: &[u8]) -> anyhow::Result<Self::G2> {
        if bytes.len() != Self::G2_GPU_BYTES {
            anyhow::bail!("Invalid G2 byte length from GPU: {}", bytes.len());
        }
        if is_all_zero(&bytes[4 * Self::FQ_GPU_BYTES..6 * Self::FQ_GPU_BYTES]) {
            return Ok(Self::G2::identity());
        }

        let x_c0_be = gpu_limbs_to_be_coord(&bytes[0..Self::FQ_GPU_BYTES]);
        let x_c1_be = gpu_limbs_to_be_coord(
            &bytes[Self::FQ_GPU_BYTES..2 * Self::FQ_GPU_BYTES],
        );
        let y_c0_be = gpu_limbs_to_be_coord(
            &bytes[2 * Self::FQ_GPU_BYTES..3 * Self::FQ_GPU_BYTES],
        );
        let y_c1_be = gpu_limbs_to_be_coord(
            &bytes[3 * Self::FQ_GPU_BYTES..4 * Self::FQ_GPU_BYTES],
        );

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

    fn serialize_scalar(s: &Self::Scalar) -> Vec<u8> {
        scalar_to_le_bytes(s).to_vec()
    }

    fn deserialize_scalar(bytes: &[u8]) -> anyhow::Result<Self::Scalar> {
        let mut arr = [0u8; 32];
        arr.copy_from_slice(bytes);
        let scalar: Option<blstrs::Scalar> =
            blstrs::Scalar::from_bytes_le(&arr).into();
        scalar.ok_or_else(|| anyhow::anyhow!("Invalid scalar bytes from GPU"))
    }

    fn scalar_to_windows(s: &Self::Scalar, c: usize) -> Vec<u32> {
        let bytes = scalar_to_le_bytes(s);
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

    fn g2_bucket_width() -> usize {
        8
    }

    fn glv_bucket_width() -> usize {
        13
    }

    fn g1_msm_bucket_width(n: usize) -> usize {
        if n < 256 {
            return 13;
        }
        let effective_n = 2 * n;
        let mut best_c = 13;
        let mut best_cost = u64::MAX;
        for c in 10..=13 {
            let windows = 128u64.div_ceil(c as u64);
            let buckets = 1u64 << (c - 1);
            let cost = windows * (effective_n as u64 + buckets);
            if cost < best_cost {
                best_cost = cost;
                best_c = c;
            }
        }
        best_c
    }

    fn decompose_g1_msm_scalar(
        s: &Self::Scalar,
        c: usize,
    ) -> G1MsmDecomposition {
        let (k1, k1_neg, k2, k2_neg) = crate::glv::bls12_381::glv_decompose(s);
        G1MsmDecomposition::Glv {
            k1_windows: crate::glv::bls12_381::u128_to_signed_windows(k1, c),
            k1_neg,
            k2_windows: crate::glv::bls12_381::u128_to_signed_windows(k2, c),
            k2_neg,
        }
    }

    fn decompose_g1_msm_scalar_glv_windows(
        s: &Self::Scalar,
        c: usize,
    ) -> Option<GlvWindowDecomposition> {
        let (k1, k1_neg, k2, k2_neg) = crate::glv::bls12_381::glv_decompose(s);
        Some((
            crate::glv::bls12_381::u128_to_signed_windows(k1, c),
            k1_neg,
            crate::glv::bls12_381::u128_to_signed_windows(k2, c),
            k2_neg,
        ))
    }

    fn g1_endomorphism_base_bytes(base_bytes: &[u8]) -> Option<Vec<u8>> {
        Some(crate::glv::bls12_381::endomorphism_g1_bytes(base_bytes).to_vec())
    }

    fn negate_g1_base_bytes(point_bytes: &mut [u8]) {
        crate::glv::bls12_381::negate_g1_bytes(point_bytes);
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
