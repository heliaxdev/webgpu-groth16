use std::ops::{Add, Mul, Sub};

use ff::{Field, PrimeField};

pub trait GpuCurve: 'static {
    type Scalar: PrimeField;
    type G1Affine: Clone + std::fmt::Debug;
    type G2Affine: Clone + std::fmt::Debug;
    type G1Projective: Clone;
    type G2Projective: Clone;

    const NTT_SOURCE: &'static str;
    const MSM_SOURCE: &'static str;

    // Serialization
    fn serialize_g1(point: &Self::G1Affine) -> Vec<u8>;
    fn deserialize_scalar(bytes: &[u8]) -> anyhow::Result<Self::Scalar>;

    // CPU Fallbacks & Math
    fn msm_g2_cpu(bases: &[Self::G2Affine], scalars: &[Self::Scalar]) -> Self::G2Projective;
    fn msm_g1_cpu(bases: &[Self::G1Affine], scalars: &[Self::Scalar]) -> Self::G1Projective;

    fn affine_to_proj_g1(p: &Self::G1Affine) -> Self::G1Projective;
    fn affine_to_proj_g2(p: &Self::G2Affine) -> Self::G2Projective;
    fn proj_to_affine_g1(p: &Self::G1Projective) -> Self::G1Affine;
    fn proj_to_affine_g2(p: &Self::G2Projective) -> Self::G2Affine;

    fn add_g1_proj(a: &Self::G1Projective, b: &Self::G1Projective) -> Self::G1Projective;
    fn sub_g1_proj(a: &Self::G1Projective, b: &Self::G1Projective) -> Self::G1Projective;
    fn add_g2_proj(a: &Self::G2Projective, b: &Self::G2Projective) -> Self::G2Projective;

    fn mul_g1_scalar(a: &Self::G1Affine, b: &Self::Scalar) -> Self::G1Projective;
    fn mul_g2_scalar(a: &Self::G2Affine, b: &Self::Scalar) -> Self::G2Projective;
    fn mul_g1_proj_scalar(a: &Self::G1Projective, b: &Self::Scalar) -> Self::G1Projective;
}

pub enum Bls12 {}

impl GpuCurve for Bls12 {
    type Scalar = blstrs::Scalar;
    type G1Affine = blstrs::G1Affine;
    type G2Affine = blstrs::G2Affine;
    type G1Projective = blstrs::G1Projective;
    type G2Projective = blstrs::G2Projective;

    const NTT_SOURCE: &'static str = concat!(
        include_str!("../shader/bls12_381/fr.wgsl"),
        "\n",
        include_str!("../shader/bls12_381/fp.wgsl"),
        "\n",
        include_str!("../shader/bls12_381/ntt.wgsl"),
    );

    const MSM_SOURCE: &'static str = concat!(
        include_str!("../shader/bls12_381/fp.wgsl"),
        "\n",
        include_str!("../shader/bls12_381/curve.wgsl"),
        "\n",
        include_str!("../shader/bls12_381/msm.wgsl"),
    );

    fn serialize_g1(point: &Self::G1Affine) -> Vec<u8> {
        let uncompressed = point.to_uncompressed();

        // Reverse Big-Endian 48-byte chunks to Little-Endian for WGSL
        let mut x = uncompressed[0..48].to_vec();
        x.reverse();

        let mut y = uncompressed[48..96].to_vec();
        y.reverse();

        // Z = 1 for Jacobian Affine injection
        let mut z = vec![0u8; 48];
        z[0] = 1;

        let mut wgsl_bytes = Vec::with_capacity(144);
        wgsl_bytes.extend_from_slice(&x);
        wgsl_bytes.extend_from_slice(&y);
        wgsl_bytes.extend_from_slice(&z);
        wgsl_bytes
    }

    fn deserialize_scalar(bytes: &[u8]) -> anyhow::Result<Self::Scalar> {
        let mut arr = [0u8; 32];
        arr.copy_from_slice(bytes);
        // FIXME: blstrs::Scalar::from_bytes_le returns CtOption, we convert it to
        // Option then Result. to prevent sidechannel attacks, we must preserve CtOption
        let scalar: Option<blstrs::Scalar> = blstrs::Scalar::from_bytes_le(&arr).into();
        scalar.ok_or_else(|| anyhow::anyhow!("Invalid scalar bytes from GPU"))
    }

    fn msm_g2_cpu(bases: &[Self::G2Affine], scalars: &[Self::Scalar]) -> Self::G2Projective {
        let mut result = blstrs::G2Projective::identity();
        for (base, scalar) in bases.iter().zip(scalars.iter()) {
            result = result.add(base.mul(scalar));
        }
        result
    }

    fn msm_g1_cpu(bases: &[Self::G1Affine], scalars: &[Self::Scalar]) -> Self::G1Projective {
        let mut result = blstrs::G1Projective::identity();
        for (base, scalar) in bases.iter().zip(scalars.iter()) {
            result = result.add(base.mul(scalar));
        }
        result
    }

    fn affine_to_proj_g1(p: &Self::G1Affine) -> Self::G1Projective {
        p.into()
    }
    fn affine_to_proj_g2(p: &Self::G2Affine) -> Self::G2Projective {
        p.into()
    }
    fn proj_to_affine_g1(p: &Self::G1Projective) -> Self::G1Affine {
        p.into()
    }
    fn proj_to_affine_g2(p: &Self::G2Projective) -> Self::G2Affine {
        p.into()
    }

    fn add_g1_proj(a: &Self::G1Projective, b: &Self::G1Projective) -> Self::G1Projective {
        a.add(b)
    }
    fn sub_g1_proj(a: &Self::G1Projective, b: &Self::G1Projective) -> Self::G1Projective {
        a.sub(b)
    }
    fn add_g2_proj(a: &Self::G2Projective, b: &Self::G2Projective) -> Self::G2Projective {
        a.add(b)
    }

    fn mul_g1_scalar(a: &Self::G1Affine, b: &Self::Scalar) -> Self::G1Projective {
        a.mul(b)
    }
    fn mul_g2_scalar(a: &Self::G2Affine, b: &Self::Scalar) -> Self::G2Projective {
        a.mul(b)
    }
    fn mul_g1_proj_scalar(a: &Self::G1Projective, b: &Self::Scalar) -> Self::G1Projective {
        a.mul(b)
    }
}
