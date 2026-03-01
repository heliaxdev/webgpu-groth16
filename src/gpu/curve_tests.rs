use super::{FQ_GPU_BYTES, FQ_GPU_PADDED_BYTES, G1_GPU_BYTES, G2_GPU_BYTES, GpuCurve};
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
        let parsed = <Bls12 as GpuCurve>::deserialize_g1(&bytes).expect("random g1 deserialize");
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
        let parsed = <Bls12 as GpuCurve>::deserialize_g2(&bytes).expect("random g2 deserialize");
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
        let parsed = <Bls12 as GpuCurve>::deserialize_scalar(&bytes).expect("scalar deserialize");
        assert_eq!(parsed, *s, "scalar roundtrip failed for test case {i}");
    }
}

/// G1 serialization byte layout: x(120+8pad) || y(120+8pad) || z(120+8pad).
#[test]
fn g1_serialization_byte_layout() {
    let g = G1Affine::generator();
    let bytes = <Bls12 as GpuCurve>::serialize_g1(&g);

    assert_eq!(bytes.len(), G1_GPU_BYTES);

    // x and y should not be all zeros for the generator
    assert!(!bytes[0..FQ_GPU_BYTES].iter().all(|&b| b == 0));
    let y_start = FQ_GPU_PADDED_BYTES;
    assert!(
        !bytes[y_start..y_start + FQ_GPU_BYTES]
            .iter()
            .all(|&b| b == 0)
    );

    // z = 1 in 13-bit format: limb[0] = 1, rest zeros (including padding)
    let z_start = 2 * FQ_GPU_PADDED_BYTES;
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
