use std::ops::{Add, Mul, Sub};

use ff::{Field, PrimeField, PrimeFieldBits};
use group::Group;

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

    // Optimal bucket width for MSM (c such that 2^c buckets)
    fn bucket_width() -> usize;

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
        include_str!("../shader/bls12_381/fp.wgsl"),
        "\n",
        include_str!("../shader/bls12_381/curve.wgsl"),
        "\n",
        include_str!("../shader/bls12_381/msm.wgsl"),
    );

    const POLY_OPS_SOURCE: &'static str = concat!(
        include_str!("../shader/bls12_381/fr.wgsl"),
        "\n",
        include_str!("../shader/bls12_381/poly_ops.wgsl"),
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

    fn deserialize_g1(bytes: &[u8]) -> anyhow::Result<Self::G1> {
        let mut x = [0u8; 48];
        let mut y = [0u8; 48];

        x.copy_from_slice(&bytes[0..48]);
        y.copy_from_slice(&bytes[48..96]);

        let mut x_be = x;
        x_be.reverse();
        let mut y_be = y;
        y_be.reverse();

        let mut uncompressed = [0u8; 96];
        uncompressed[..48].copy_from_slice(&x_be);
        uncompressed[48..].copy_from_slice(&y_be);

        let ct: subtle::CtOption<blstrs::G1Affine> =
            blstrs::G1Affine::from_uncompressed(&uncompressed);
        let affine: Option<blstrs::G1Affine> = Option::from(ct);
        if let Some(affine) = affine {
            return Ok(affine.into());
        }

        anyhow::bail!("Failed to deserialize G1 point from GPU")
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

            for j in 0..c.div_ceil(8).min(4) {
                let byte_idx = bit_offset / 8 + j;
                if byte_idx < 32 {
                    window |= (bytes[byte_idx] as u64) << (j * 8);
                }
            }

            let mask = (1u64 << c) - 1;
            windows.push((window & mask) as u32);
        }

        windows
    }

    fn bucket_width() -> usize {
        15
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

    fn serialize_g2(point: &Self::G2Affine) -> Vec<u8> {
        let uncompressed = point.to_uncompressed();

        // blstrs G2 uncompressed layout (192 bytes):
        // x.c1 (48 bytes, BE), x.c0 (48 bytes, BE)
        // y.c1 (48 bytes, BE), y.c0 (48 bytes, BE)
        // WGSL expects (288 bytes total):
        // x.c0, x.c1, y.c0, y.c1, z.c0, z.c1 (each 48 bytes, LE)

        let mut x_c1 = uncompressed[0..48].to_vec();
        x_c1.reverse();
        let mut x_c0 = uncompressed[48..96].to_vec();
        x_c0.reverse();
        let mut y_c1 = uncompressed[96..144].to_vec();
        y_c1.reverse();
        let mut y_c0 = uncompressed[144..192].to_vec();
        y_c0.reverse();

        // Z = 1 (z.c0 = 1, z.c1 = 0)
        let mut z_c0 = vec![0u8; 48];
        z_c0[0] = 1;
        let z_c1 = vec![0u8; 48];

        let mut wgsl_bytes = Vec::with_capacity(288);
        wgsl_bytes.extend_from_slice(&x_c0);
        wgsl_bytes.extend_from_slice(&x_c1);
        wgsl_bytes.extend_from_slice(&y_c0);
        wgsl_bytes.extend_from_slice(&y_c1);
        wgsl_bytes.extend_from_slice(&z_c0);
        wgsl_bytes.extend_from_slice(&z_c1);
        wgsl_bytes
    }

    fn deserialize_g2(bytes: &[u8]) -> anyhow::Result<Self::G2> {
        let mut x_c0 = [0u8; 48];
        x_c0.copy_from_slice(&bytes[0..48]);
        x_c0.reverse();
        let mut x_c1 = [0u8; 48];
        x_c1.copy_from_slice(&bytes[48..96]);
        x_c1.reverse();
        let mut y_c0 = [0u8; 48];
        y_c0.copy_from_slice(&bytes[96..144]);
        y_c0.reverse();
        let mut y_c1 = [0u8; 48];
        y_c1.copy_from_slice(&bytes[144..192]);
        y_c1.reverse();

        let mut uncompressed = [0u8; 192];
        uncompressed[0..48].copy_from_slice(&x_c1);
        uncompressed[48..96].copy_from_slice(&x_c0);
        uncompressed[96..144].copy_from_slice(&y_c1);
        uncompressed[144..192].copy_from_slice(&y_c0);

        let ct: subtle::CtOption<blstrs::G2Affine> =
            blstrs::G2Affine::from_uncompressed(&uncompressed);
        let affine: Option<blstrs::G2Affine> = Option::from(ct);
        if let Some(affine) = affine {
            return Ok(affine.into());
        }

        anyhow::bail!("Failed to deserialize G2 point from GPU")
    }

    fn g2_identity() -> Self::G2 {
        Self::G2::identity()
    }
}
