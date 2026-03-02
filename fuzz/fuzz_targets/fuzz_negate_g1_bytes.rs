#![no_main]

use libfuzzer_sys::fuzz_target;
use blstrs::{G1Affine, G1Projective, Scalar};
use ff::PrimeField;
use group::{Curve, Group};
use webgpu_groth16::glv::negate_g1_bytes;
use webgpu_groth16::gpu::curve::GpuCurve;

fuzz_target!(|data: [u8; 32]| {
    let Some(s) = Scalar::from_repr_vartime(data) else {
        return;
    };

    let point: G1Affine = (G1Projective::generator() * s).to_affine();
    let serialized = <blstrs::Bls12 as GpuCurve>::serialize_g1(&point);
    let mut bytes = serialized.clone();

    // Property 1: double negation is identity
    negate_g1_bytes(&mut bytes);
    negate_g1_bytes(&mut bytes);
    assert_eq!(
        bytes, serialized,
        "double negation should return the original bytes"
    );

    // Property 2: negated bytes deserialize to -P
    let mut neg_bytes = serialized.clone();
    negate_g1_bytes(&mut neg_bytes);

    let neg_deserialized = <blstrs::Bls12 as GpuCurve>::deserialize_g1(&neg_bytes)
        .expect("negated point should deserialize");
    let neg_point = -G1Projective::from(point);
    assert_eq!(
        neg_deserialized, neg_point,
        "negated bytes should deserialize to -P"
    );
});
