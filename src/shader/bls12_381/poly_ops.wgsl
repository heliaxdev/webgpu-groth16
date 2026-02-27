// src/shader/bls12_381/poly_ops.wgsl

// Helper for F_r modular subtraction (handles underflow safely)
fn sub_fr(a: U256, b: U256) -> U256 {
    var is_less = false;
    for (var i = 7u; i < 8u; i = i - 1u) {
        if a.limbs[i] < b.limbs[i] { is_less = true; break; }
        if a.limbs[i] > b.limbs[i] { break; }
        if i == 0u { break; }
    }

    var diff = sub_u256(a, b);
    if is_less {
        diff = add_u256(diff, U256(R_MODULUS));
    }
    return diff;
}

// ============================================================================
// COSET SHIFT PIPELINE
// ============================================================================
@group(0) @binding(0) var<storage, read_write> shift_data: array<U256>;
@group(0) @binding(1) var<storage, read> shift_factors: array<U256>;

@compute @workgroup_size(256)
fn coset_shift(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&shift_factors) { return; }

    shift_data[i] = mul_montgomery_u256(shift_data[i], shift_factors[i]);
}

// ============================================================================
// POINTWISE POLYNOMIAL MATH PIPELINE
// H[i] = (A[i] * B[i] - C[i]) / Z_H[i]
// ============================================================================
@group(0) @binding(0) var<storage, read> A: array<U256>;
@group(0) @binding(1) var<storage, read> B: array<U256>;
@group(0) @binding(2) var<storage, read> C: array<U256>;
@group(0) @binding(3) var<storage, read_write> H: array<U256>;
@group(0) @binding(4) var<storage, read> Z_invs: array<U256>; // [z_even_inv, z_odd_inv]

@compute @workgroup_size(256)
fn pointwise_poly(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&A) { return; }

    let a = A[i];
    let b = B[i];
    let c = C[i];

    let ab = mul_montgomery_u256(a, b);
    let res = sub_fr(ab, c);

    var z_inv: U256;
    if (i % 2u) == 0u {
        z_inv = Z_invs[0];
    } else {
        z_inv = Z_invs[1];
    }

    H[i] = mul_montgomery_u256(res, z_inv);
}

// ============================================================================
// MONTGOMERY DOMAIN BRIDGES
// ============================================================================

// R^2 mod r for the BLS12-381 scalar field (F_r).
// Required to convert Standard Form scalars into Montgomery Form.
const R2_MOD_R = U256(array<u32, 8>(
    0xc422c543u, 0x86dc10c2u, 0x4a1168bau, 0xb0e50a5du,
    0x28981180u, 0x49d9f8e5u, 0xa060e40du, 0x099d2609u
));

fn to_montgomery_u256(a: U256) -> U256 {
    return mul_montgomery_u256(a, R2_MOD_R);
}

fn from_montgomery_u256(a: U256) -> U256 {
    let one = U256(array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u));
    return mul_montgomery_u256(a, one);
}

@group(0) @binding(0) var<storage, read_write> mont_buf: array<U256>;

@compute @workgroup_size(256)
fn to_montgomery_array(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&mont_buf) { return; }
    mont_buf[i] = to_montgomery_u256(mont_buf[i]);
}

@compute @workgroup_size(256)
fn from_montgomery_array(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&mont_buf) { return; }
    mont_buf[i] = from_montgomery_u256(mont_buf[i]);
}
