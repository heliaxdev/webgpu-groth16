#![no_main]

use libfuzzer_sys::fuzz_target;
use webgpu_groth16::gpu::curve::{fq_bytes_to_13bit, fq_13bit_to_bytes};

fuzz_target!(|data: [u8; 48]| {
    // Round-trip: 48-byte LE Fq → 120-byte 13-bit limbs → 48-byte LE Fq
    let limbs = fq_bytes_to_13bit(&data);
    assert_eq!(limbs.len(), 120);

    // Verify each limb fits in 13 bits
    for i in 0..30 {
        let limb = u32::from_le_bytes([
            limbs[i * 4],
            limbs[i * 4 + 1],
            limbs[i * 4 + 2],
            limbs[i * 4 + 3],
        ]);
        assert!(limb < (1 << 13), "limb {i} = {limb} exceeds 13 bits");
    }

    let recovered = fq_13bit_to_bytes(&limbs);
    assert_eq!(recovered.len(), 48);

    // The round-trip must be exact for values that fit in 384 bits.
    // Since we pack 30×13 = 390 bits but Fq is only 381 bits, the top bits
    // of the last limb may get truncated. The round-trip is still exact for
    // all 48-byte inputs because the packing/unpacking is lossless on the
    // 384-bit input domain (384 < 390).
    assert_eq!(
        &recovered[..],
        &data[..],
        "fq_13bit round-trip failed"
    );
});
