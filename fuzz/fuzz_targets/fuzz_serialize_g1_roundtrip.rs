#![no_main]

use libfuzzer_sys::fuzz_target;
use blstrs::{G1Affine, G1Projective, Scalar};
use ff::PrimeField;
use group::{Curve, Group};
use webgpu_groth16::gpu::curve::GpuCurve;

fuzz_target!(|data: [u8; 32]| {
    // Generate a valid G1 point by scalar-multiplying the generator.
    // This guarantees the point is on the curve and in the correct subgroup.
    let Some(s) = Scalar::from_repr_vartime(data) else {
        return;
    };

    let point: G1Affine = (G1Projective::generator() * s).to_affine();
    let serialized = <blstrs::Bls12 as GpuCurve>::serialize_g1(&point);
    assert_eq!(serialized.len(), 384, "G1 GPU bytes should be 384");

    let deserialized = <blstrs::Bls12 as GpuCurve>::deserialize_g1(&serialized)
        .expect("round-trip deserialization should succeed");

    // Compare in projective coordinates
    let expected = G1Projective::from(point);
    assert_eq!(
        deserialized, expected,
        "G1 serialize/deserialize round-trip failed"
    );
});
