// src/shader/bls12_381/fp.wgsl

// ============================================================================
// BASE FIELD (F_q) - 384-bit arithmetic
// Used for Elliptic Curve point additions and MSMs.
// ============================================================================

struct U384 {
    limbs: array<u32, 12>,
}

fn add_u384(a: U384, b: U384) -> U384 {
    var result: U384;
    var carry: u32 = 0u;

    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        let a_val = a.limbs[i];
        let b_val = b.limbs[i];

        let sum1 = a_val + b_val;
        let carry1 = u32(sum1 < a_val);

        let sum2 = sum1 + carry;
        let carry2 = u32(sum2 < sum1);

        result.limbs[i] = sum2;
        carry = carry1 + carry2;
    }

    return result;
}

fn sub_u384(a: U384, b: U384) -> U384 {
    var result: U384;
    var borrow: u32 = 0u;

    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        let a_val = a.limbs[i];
        let b_val = b.limbs[i];

        let diff1 = a_val - b_val;
        let borrow1 = u32(a_val < b_val);

        let diff2 = diff1 - borrow;
        let borrow2 = u32(diff1 < borrow);

        result.limbs[i] = diff2;
        borrow = borrow1 + borrow2;
    }

    return result;
}

// F_q Modular Addition
fn add_mod_q(a: U384, b: U384) -> U384 {
    var sum: U384;
    var carry: u32 = 0u;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        let a_val = a.limbs[i];
        let b_val = b.limbs[i];

        let sum1 = a_val + b_val;
        let carry1 = u32(sum1 < a_val);

        let sum2 = sum1 + carry;
        let carry2 = u32(sum2 < sum1);

        sum.limbs[i] = sum2;
        carry = carry1 + carry2;
    }

    var is_gte = true;
    for (var i = 11u; i < 12u; i = i - 1u) {
        if sum.limbs[i] > Q_MODULUS[i] { break; }
        if sum.limbs[i] < Q_MODULUS[i] { is_gte = false; break; }
        if i == 0u { break; }
    }
    if carry > 0u || is_gte {
        sum = sub_u384(sum, U384(Q_MODULUS));
    }
    return sum;
}

// F_q Modular Subtraction
fn sub_mod_q(a: U384, b: U384) -> U384 {
    var is_less = false;
    for (var i = 11u; i < 12u; i = i - 1u) {
        if a.limbs[i] < b.limbs[i] { is_less = true; break; }
        if a.limbs[i] > b.limbs[i] { break; }
        if i == 0u { break; }
    }
    var diff = sub_u384(a, b);
    if is_less {
        // Underflow occurred, wrap around strictly modulo Q
        diff = add_u384(diff, U384(Q_MODULUS));
    }
    return diff;
}

// ============================================================================
// BLS12-381 CONSTANTS
// ============================================================================

// Scalar Field (F_r) Modulus (256 bits, 8 limbs)
const R_MODULUS = array<u32, 8>(
    0x00000001u, 0xffffffffu, 0xfffe5bfeu, 0x53bda402u,
    0x09a1d805u, 0x3339d808u, 0x299d7d48u, 0x73eda753u
);

// Base Field (F_q) Modulus (384 bits, 12 limbs)
const Q_MODULUS = array<u32, 12>(
    0xffffaaabu, 0xb9feffffu, 0xb153ffffu, 0x1eabfffeu,
    0xf6b0f624u, 0x6730d2a0u, 0xf38512bfu, 0x64774b84u,
    0x434bacd7u, 0x4b1ba7b6u, 0x397fe69au, 0x1a0111eau
);

// Montgomery constants: -r^{-1} mod 2^32 and -q^{-1} mod 2^32
const INV_R: u32 = 0xffffffffu;
const INV_Q: u32 = 0xfffcfffdu;

// ============================================================================
// 64-BIT MULTIPLICATION UTILITY
// ============================================================================

// Simulates a 32x32 -> 64-bit multiplication.
// Returns a vec2<u32> where x is the low 32 bits and y is the high 32 bits.
fn mul_u32(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;

    let carry1 = p0 >> 16u;
    let sum1 = p1 + carry1;
    let p1_lo = sum1 & 0xFFFFu;
    let p1_hi = sum1 >> 16u;

    let sum2 = p2 + p1_lo;
    let p2_lo = sum2 & 0xFFFFu;
    let p2_hi = sum2 >> 16u;

    let lo = (p2_lo << 16u) | (p0 & 0xFFFFu);
    let hi = p3 + p1_hi + p2_hi;

    return vec2<u32>(lo, hi);
}

// ============================================================================
// SCALAR FIELD MONTGOMERY MULTIPLICATION (F_r)
// ============================================================================

fn mul_montgomery_u256(a: U256, b: U256) -> U256 {
    // CIOS Montgomery multiplication for F_r (n=8, t shrinks from 16 to 10)
    var t = array<u32, 10>(
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u
    );

    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let ai = a.limbs[i];

        // Multiply: accumulate a[i] * b[j] into t
        var C: u32 = 0u;
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            let prod = mul_u32(ai, b.limbs[j]);
            let sum1 = t[j] + prod.x;
            let c1 = u32(sum1 < t[j]);
            let sum2 = sum1 + C;
            let c2 = u32(sum2 < sum1);
            t[j] = sum2;
            C = prod.y + c1 + c2;
        }
        let add_n = t[8u] + C;
        let cn = u32(add_n < t[8u]);
        t[8u] = add_n;
        t[9u] = t[9u] + cn;

        // Reduce: m * r with built-in shift
        let m = t[0u] * INV_R;

        let prod0 = mul_u32(m, R_MODULUS[0u]);
        let s0 = t[0u] + prod0.x;
        let c0 = u32(s0 < t[0u]);
        C = prod0.y + c0;

        for (var j: u32 = 1u; j < 8u; j = j + 1u) {
            let prod = mul_u32(m, R_MODULUS[j]);
            let sum1 = t[j] + prod.x;
            let c1 = u32(sum1 < t[j]);
            let sum2 = sum1 + C;
            let c2 = u32(sum2 < sum1);
            t[j - 1u] = sum2;
            C = prod.y + c1 + c2;
        }
        let add_n2 = t[8u] + C;
        let cn2 = u32(add_n2 < t[8u]);
        t[7u] = add_n2;
        t[8u] = t[9u] + cn2;
        t[9u] = 0u;
    }

    var result: U256;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        result.limbs[i] = t[i];
    }

    var is_gte = true;
    if t[8u] > 0u {
        is_gte = true;
    } else {
        for (var i: u32 = 7u; i < 8u; i = i - 1u) {
            if result.limbs[i] > R_MODULUS[i] { break; }
            if result.limbs[i] < R_MODULUS[i] { is_gte = false; break; }
            if i == 0u { break; }
        }
    }
    if is_gte {
        result = sub_u256(result, U256(R_MODULUS));
    }
    return result;
}

// ============================================================================
// BASE FIELD MONTGOMERY MULTIPLICATION (F_q)
// ============================================================================

fn mul_montgomery_u384(a: U384, b: U384) -> U384 {
    // CIOS (Coarsely Integrated Operand Scanning) Montgomery multiplication.
    // Interleaves multiply and reduce in each outer iteration with a built-in
    // shift, reducing the accumulator from 2n=24 to n+2=14 limbs.
    var t = array<u32, 14>(
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u
    );

    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        let ai = a.limbs[i];

        // Multiply: accumulate a[i] * b[j] into t
        var C: u32 = 0u;
        for (var j: u32 = 0u; j < 12u; j = j + 1u) {
            let prod = mul_u32(ai, b.limbs[j]);
            let sum1 = t[j] + prod.x;
            let c1 = u32(sum1 < t[j]);
            let sum2 = sum1 + C;
            let c2 = u32(sum2 < sum1);
            t[j] = sum2;
            C = prod.y + c1 + c2;
        }
        let add_n = t[12u] + C;
        let cn = u32(add_n < t[12u]);
        t[12u] = add_n;
        t[13u] = t[13u] + cn;

        // Reduce: m * q with built-in shift (write to j-1)
        let m = t[0u] * INV_Q;

        // j=0: t[0] + m*q[0] is zero by construction, just compute carry
        let prod0 = mul_u32(m, Q_MODULUS[0u]);
        let s0 = t[0u] + prod0.x;
        let c0 = u32(s0 < t[0u]);
        C = prod0.y + c0;

        // j=1..11: accumulate and shift (write to j-1)
        for (var j: u32 = 1u; j < 12u; j = j + 1u) {
            let prod = mul_u32(m, Q_MODULUS[j]);
            let sum1 = t[j] + prod.x;
            let c1 = u32(sum1 < t[j]);
            let sum2 = sum1 + C;
            let c2 = u32(sum2 < sum1);
            t[j - 1u] = sum2;
            C = prod.y + c1 + c2;
        }
        // Shift the top words
        let add_n2 = t[12u] + C;
        let cn2 = u32(add_n2 < t[12u]);
        t[11u] = add_n2;
        t[12u] = t[13u] + cn2;
        t[13u] = 0u;
    }

    // Result is in t[0..11]
    var result: U384;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        result.limbs[i] = t[i];
    }

    // Conditional subtraction if result >= Q
    var is_gte = true;
    if t[12u] > 0u {
        is_gte = true;
    } else {
        for (var i: u32 = 11u; i < 12u; i = i - 1u) {
            if result.limbs[i] > Q_MODULUS[i] { break; }
            if result.limbs[i] < Q_MODULUS[i] { is_gte = false; break; }
            if i == 0u { break; }
        }
    }
    if is_gte {
        result = sub_u384(result, U384(Q_MODULUS));
    }
    return result;
}
