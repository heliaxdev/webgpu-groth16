#![no_main]

use libfuzzer_sys::fuzz_target;
use blstrs::Scalar;
use ff::PrimeField;
use webgpu_groth16::glv::glv_decompose;

/// λ = endomorphism eigenvalue in Fr (BLS12-381).
fn lambda() -> Scalar {
    Scalar::from_repr_vartime([
        0x01, 0x00, 0x00, 0x00, 0xfe, 0xff, 0xff, 0xff, 0xfc, 0xb7, 0xfc, 0xff, 0x01, 0x00,
        0x78, 0xa7, 0x04, 0xd8, 0xa1, 0x09, 0x08, 0xd8, 0x39, 0x33, 0x48, 0x7d, 0x9d, 0x29,
        0x53, 0xa7, 0xed, 0x73,
    ])
    .expect("LAMBDA is a valid scalar")
}

fn scalar_from_u128(val: u128, negative: bool) -> Scalar {
    let lo = val as u64;
    let hi = (val >> 64) as u64;
    let mut bytes = [0u8; 32];
    bytes[0..8].copy_from_slice(&lo.to_le_bytes());
    bytes[8..16].copy_from_slice(&hi.to_le_bytes());
    let s = Scalar::from_repr_vartime(bytes).expect("128-bit value is a valid scalar");
    if negative { -s } else { s }
}

fuzz_target!(|data: [u8; 32]| {
    // Try to parse as a valid scalar; skip invalid inputs
    let Some(k) = Scalar::from_repr_vartime(data) else {
        return;
    };

    let (k1_abs, k1_neg, k2_abs, k2_neg) = glv_decompose(&k);

    // Property 1: both halves fit in ~128 bits
    assert!(
        k1_abs < (1u128 << 127) + (1u128 << 126),
        "k1 too large: {k1_abs}"
    );
    assert!(
        k2_abs < (1u128 << 127) + (1u128 << 126),
        "k2 too large: {k2_abs}"
    );

    // Property 2: k1 + k2·λ ≡ k (mod r)
    let k1_scalar = scalar_from_u128(k1_abs, k1_neg);
    let k2_scalar = scalar_from_u128(k2_abs, k2_neg);
    let reconstructed = k1_scalar + k2_scalar * lambda();
    assert_eq!(
        reconstructed, k,
        "GLV decomposition failed: k1={k1_abs} (neg={k1_neg}), k2={k2_abs} (neg={k2_neg})"
    );
});
