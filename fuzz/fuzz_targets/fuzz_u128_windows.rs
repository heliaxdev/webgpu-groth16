#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use webgpu_groth16::glv::{u128_to_windows, u128_to_signed_windows};

#[derive(Arbitrary, Debug)]
struct Input {
    value: u128,
    /// Window width, will be clamped to [8, 16]
    c_raw: u8,
}

fuzz_target!(|input: Input| {
    let c = (input.c_raw % 9) as usize + 8; // c ∈ [8, 16]
    let value = input.value;

    // === Unsigned windows ===
    let windows = u128_to_windows(value, c);
    let num_windows = 128_usize.div_ceil(c);
    assert_eq!(windows.len(), num_windows, "wrong number of unsigned windows");

    // Each window value must fit in c bits
    let mask = (1u32 << c) - 1;
    for (i, &w) in windows.iter().enumerate() {
        assert!(w <= mask, "unsigned window {i} = {w} exceeds {c} bits");
    }

    // Reconstruction: Σ(w[i] · 2^(i·c)) == value
    let mut reconstructed = 0u128;
    for (i, &w) in windows.iter().enumerate() {
        let shift = i * c;
        if shift < 128 {
            reconstructed |= (w as u128) << shift;
        }
    }
    assert_eq!(
        reconstructed, value,
        "unsigned window reconstruction failed for value={value:#x} c={c}"
    );

    // === Signed windows ===
    let signed = u128_to_signed_windows(value, c);
    let half = 1u32 << (c - 1);

    // Each absolute value must be ≤ half
    for (i, &(abs, _neg)) in signed.iter().enumerate() {
        assert!(
            abs <= half,
            "signed window {i}: abs={abs} exceeds half={half}"
        );
    }

    // Signed reconstruction: Σ(±|w[i]| · 2^(i·c)) == value (mod 2^128)
    // We compute in i128 to handle signed arithmetic, wrapping at 128 bits.
    let mut signed_reconstructed = 0i128;
    for (i, &(abs, neg)) in signed.iter().enumerate() {
        let shift = i * c;
        if shift < 128 {
            let val = (abs as i128) << shift;
            if neg {
                signed_reconstructed = signed_reconstructed.wrapping_sub(val);
            } else {
                signed_reconstructed = signed_reconstructed.wrapping_add(val);
            }
        }
    }
    assert_eq!(
        signed_reconstructed as u128, value,
        "signed window reconstruction failed for value={value:#x} c={c}"
    );
});
