// src/shader/bls12_381/fp.wgsl

// ============================================================================
// BASE FIELD (F_q) - 384-bit arithmetic using 30 × 13-bit limbs
// Each limb is stored in a u32 but only uses the bottom 13 bits (0..8191).
// This representation eliminates the need for mul_u32 (16-bit decomposition)
// since 13×13 = 26 bits fits natively in a u32 multiply.
// R = 2^390 for Montgomery form.
// ============================================================================

struct U384 {
    limbs: array<u32, 30>,
}

fn add_u384(a: U384, b: U384) -> U384 {
    var result: U384;
    var carry: u32 = 0u;
    for (var i: u32 = 0u; i < 30u; i = i + 1u) {
        let sum = a.limbs[i] + b.limbs[i] + carry;
        result.limbs[i] = sum & 0x1FFFu;
        carry = sum >> 13u;
    }
    return result;
}

fn sub_u384(a: U384, b: U384) -> U384 {
    var result: U384;
    var borrow: u32 = 0u;
    for (var i: u32 = 0u; i < 30u; i = i + 1u) {
        // Add 2^13 to prevent u32 underflow
        let diff = a.limbs[i] + 0x2000u - b.limbs[i] - borrow;
        result.limbs[i] = diff & 0x1FFFu;
        borrow = 1u - (diff >> 13u);
    }
    return result;
}

// F_q Modular Addition
fn add_mod_q(a: U384, b: U384) -> U384 {
    var sum = add_u384(a, b);
    // Conditional subtraction if sum >= q
    var is_gte = true;
    for (var i = 29u; i < 30u; i = i - 1u) {
        if sum.limbs[i] > Q_MODULUS[i] { break; }
        if sum.limbs[i] < Q_MODULUS[i] { is_gte = false; break; }
        if i == 0u { break; }
    }
    if is_gte {
        sum = sub_u384(sum, U384(Q_MODULUS));
    }
    return sum;
}

// F_q Modular Subtraction
fn sub_mod_q(a: U384, b: U384) -> U384 {
    var is_less = false;
    for (var i = 29u; i < 30u; i = i - 1u) {
        if a.limbs[i] < b.limbs[i] { is_less = true; break; }
        if a.limbs[i] > b.limbs[i] { break; }
        if i == 0u { break; }
    }
    var diff = sub_u384(a, b);
    if is_less {
        diff = add_u384(diff, U384(Q_MODULUS));
    }
    return diff;
}

// ============================================================================
// BLS12-381 CONSTANTS
// ============================================================================

// Scalar Field (F_r) Modulus (256 bits, 8 limbs) — unchanged (32-bit limbs)
const R_MODULUS = array<u32, 8>(
    0x00000001u, 0xffffffffu, 0xfffe5bfeu, 0x53bda402u,
    0x09a1d805u, 0x3339d808u, 0x299d7d48u, 0x73eda753u
);

// Base Field (F_q) Modulus (384 bits, 30 × 13-bit limbs)
const Q_MODULUS = array<u32, 30>(
    0x0aabu, 0x1ffdu, 0x1fffu, 0x1dffu, 0x1b9fu, 0x1fffu,
    0x054fu, 0x1fd6u, 0x0bffu, 0x00f5u, 0x1d89u, 0x0d61u,
    0x0a0fu, 0x1869u, 0x1d9cu, 0x0257u, 0x1385u, 0x1c27u,
    0x1dd2u, 0x0ec8u, 0x1acdu, 0x01a5u, 0x1ed9u, 0x0374u,
    0x1a4bu, 0x1f34u, 0x0e5fu, 0x03d4u, 0x0011u, 0x000du
);

// Montgomery constants: -r^{-1} mod 2^32 (for F_r) and -q^{-1} mod 2^13 (for F_q)
const INV_R: u32 = 0xffffffffu;
const INV_Q: u32 = 0x1ffdu;

// ============================================================================
// 64-BIT MULTIPLICATION UTILITY (used only by F_r / U256 Montgomery)
// ============================================================================

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
// SCALAR FIELD MONTGOMERY MULTIPLICATION (F_r) — unchanged (32-bit limbs)
// ============================================================================

fn mul_montgomery_u256(a: U256, b: U256) -> U256 {
    var t = array<u32, 10>(
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u
    );

    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let ai = a.limbs[i];

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

fn sqr_montgomery_u256(a: U256) -> U256 {
    return mul_montgomery_u256(a, a);
}

// ============================================================================
// BASE FIELD MONTGOMERY MULTIPLICATION (F_q) — Lazy CIOS with 13-bit limbs
//
// Scalar accumulator variables (OPT-17), literal Q constants (OPT-26).
// Eliminates array spilling and indexing overhead.
// Generated by scripts/gen_fp_unrolled.py
// ============================================================================

fn mul_montgomery_u384(a: U384, b: U384) -> U384 {
    var t0: u32 = 0u;
    var t1: u32 = 0u;
    var t2: u32 = 0u;
    var t3: u32 = 0u;
    var t4: u32 = 0u;
    var t5: u32 = 0u;
    var t6: u32 = 0u;
    var t7: u32 = 0u;
    var t8: u32 = 0u;
    var t9: u32 = 0u;
    var t10: u32 = 0u;
    var t11: u32 = 0u;
    var t12: u32 = 0u;
    var t13: u32 = 0u;
    var t14: u32 = 0u;
    var t15: u32 = 0u;
    var t16: u32 = 0u;
    var t17: u32 = 0u;
    var t18: u32 = 0u;
    var t19: u32 = 0u;
    var t20: u32 = 0u;
    var t21: u32 = 0u;
    var t22: u32 = 0u;
    var t23: u32 = 0u;
    var t24: u32 = 0u;
    var t25: u32 = 0u;
    var t26: u32 = 0u;
    var t27: u32 = 0u;
    var t28: u32 = 0u;
    var t29: u32 = 0u;
    var t30: u32 = 0u;
    var t31: u32 = 0u;

    for (var i: u32 = 0u; i < 30u; i = i + 1u) {
        let ai = a.limbs[i];

        // Phase 1: Multiply — a[i] * b[j] (unrolled)
        t0 = t0 + ai * b.limbs[0u];
        t1 = t1 + ai * b.limbs[1u];
        t2 = t2 + ai * b.limbs[2u];
        t3 = t3 + ai * b.limbs[3u];
        t4 = t4 + ai * b.limbs[4u];
        t5 = t5 + ai * b.limbs[5u];
        t6 = t6 + ai * b.limbs[6u];
        t7 = t7 + ai * b.limbs[7u];
        t8 = t8 + ai * b.limbs[8u];
        t9 = t9 + ai * b.limbs[9u];
        t10 = t10 + ai * b.limbs[10u];
        t11 = t11 + ai * b.limbs[11u];
        t12 = t12 + ai * b.limbs[12u];
        t13 = t13 + ai * b.limbs[13u];
        t14 = t14 + ai * b.limbs[14u];
        t15 = t15 + ai * b.limbs[15u];
        t16 = t16 + ai * b.limbs[16u];
        t17 = t17 + ai * b.limbs[17u];
        t18 = t18 + ai * b.limbs[18u];
        t19 = t19 + ai * b.limbs[19u];
        t20 = t20 + ai * b.limbs[20u];
        t21 = t21 + ai * b.limbs[21u];
        t22 = t22 + ai * b.limbs[22u];
        t23 = t23 + ai * b.limbs[23u];
        t24 = t24 + ai * b.limbs[24u];
        t25 = t25 + ai * b.limbs[25u];
        t26 = t26 + ai * b.limbs[26u];
        t27 = t27 + ai * b.limbs[27u];
        t28 = t28 + ai * b.limbs[28u];
        t29 = t29 + ai * b.limbs[29u];

        // Phase 2: m = t0 * (-q^{-1}) mod 2^13
        let m = (t0 * 0x1ffdu) & 0x1FFFu;

        // Phase 3: Reduce — m * q[j] (unrolled, literal constants)
        t0 = t0 + m * 0xaabu;
        t1 = t1 + m * 0x1ffdu;
        t2 = t2 + m * 0x1fffu;
        t3 = t3 + m * 0x1dffu;
        t4 = t4 + m * 0x1b9fu;
        t5 = t5 + m * 0x1fffu;
        t6 = t6 + m * 0x54fu;
        t7 = t7 + m * 0x1fd6u;
        t8 = t8 + m * 0xbffu;
        t9 = t9 + m * 0xf5u;
        t10 = t10 + m * 0x1d89u;
        t11 = t11 + m * 0xd61u;
        t12 = t12 + m * 0xa0fu;
        t13 = t13 + m * 0x1869u;
        t14 = t14 + m * 0x1d9cu;
        t15 = t15 + m * 0x257u;
        t16 = t16 + m * 0x1385u;
        t17 = t17 + m * 0x1c27u;
        t18 = t18 + m * 0x1dd2u;
        t19 = t19 + m * 0xec8u;
        t20 = t20 + m * 0x1acdu;
        t21 = t21 + m * 0x1a5u;
        t22 = t22 + m * 0x1ed9u;
        t23 = t23 + m * 0x374u;
        t24 = t24 + m * 0x1a4bu;
        t25 = t25 + m * 0x1f34u;
        t26 = t26 + m * 0xe5fu;
        t27 = t27 + m * 0x3d4u;
        t28 = t28 + m * 0x11u;
        t29 = t29 + m * 0xdu;

        // Phase 4: Carry propagation + shift (fully unrolled)
        var carry: u32 = t0 >> 13u;
        var v: u32;
        v = t1 + carry; t0 = v & 0x1FFFu; carry = v >> 13u;
        v = t2 + carry; t1 = v & 0x1FFFu; carry = v >> 13u;
        v = t3 + carry; t2 = v & 0x1FFFu; carry = v >> 13u;
        v = t4 + carry; t3 = v & 0x1FFFu; carry = v >> 13u;
        v = t5 + carry; t4 = v & 0x1FFFu; carry = v >> 13u;
        v = t6 + carry; t5 = v & 0x1FFFu; carry = v >> 13u;
        v = t7 + carry; t6 = v & 0x1FFFu; carry = v >> 13u;
        v = t8 + carry; t7 = v & 0x1FFFu; carry = v >> 13u;
        v = t9 + carry; t8 = v & 0x1FFFu; carry = v >> 13u;
        v = t10 + carry; t9 = v & 0x1FFFu; carry = v >> 13u;
        v = t11 + carry; t10 = v & 0x1FFFu; carry = v >> 13u;
        v = t12 + carry; t11 = v & 0x1FFFu; carry = v >> 13u;
        v = t13 + carry; t12 = v & 0x1FFFu; carry = v >> 13u;
        v = t14 + carry; t13 = v & 0x1FFFu; carry = v >> 13u;
        v = t15 + carry; t14 = v & 0x1FFFu; carry = v >> 13u;
        v = t16 + carry; t15 = v & 0x1FFFu; carry = v >> 13u;
        v = t17 + carry; t16 = v & 0x1FFFu; carry = v >> 13u;
        v = t18 + carry; t17 = v & 0x1FFFu; carry = v >> 13u;
        v = t19 + carry; t18 = v & 0x1FFFu; carry = v >> 13u;
        v = t20 + carry; t19 = v & 0x1FFFu; carry = v >> 13u;
        v = t21 + carry; t20 = v & 0x1FFFu; carry = v >> 13u;
        v = t22 + carry; t21 = v & 0x1FFFu; carry = v >> 13u;
        v = t23 + carry; t22 = v & 0x1FFFu; carry = v >> 13u;
        v = t24 + carry; t23 = v & 0x1FFFu; carry = v >> 13u;
        v = t25 + carry; t24 = v & 0x1FFFu; carry = v >> 13u;
        v = t26 + carry; t25 = v & 0x1FFFu; carry = v >> 13u;
        v = t27 + carry; t26 = v & 0x1FFFu; carry = v >> 13u;
        v = t28 + carry; t27 = v & 0x1FFFu; carry = v >> 13u;
        v = t29 + carry; t28 = v & 0x1FFFu; carry = v >> 13u;
        v = t30 + carry; t29 = v & 0x1FFFu; t30 = v >> 13u; t31 = 0u;
    }

    var result: U384;
    result.limbs[0u] = t0;
    result.limbs[1u] = t1;
    result.limbs[2u] = t2;
    result.limbs[3u] = t3;
    result.limbs[4u] = t4;
    result.limbs[5u] = t5;
    result.limbs[6u] = t6;
    result.limbs[7u] = t7;
    result.limbs[8u] = t8;
    result.limbs[9u] = t9;
    result.limbs[10u] = t10;
    result.limbs[11u] = t11;
    result.limbs[12u] = t12;
    result.limbs[13u] = t13;
    result.limbs[14u] = t14;
    result.limbs[15u] = t15;
    result.limbs[16u] = t16;
    result.limbs[17u] = t17;
    result.limbs[18u] = t18;
    result.limbs[19u] = t19;
    result.limbs[20u] = t20;
    result.limbs[21u] = t21;
    result.limbs[22u] = t22;
    result.limbs[23u] = t23;
    result.limbs[24u] = t24;
    result.limbs[25u] = t25;
    result.limbs[26u] = t26;
    result.limbs[27u] = t27;
    result.limbs[28u] = t28;
    result.limbs[29u] = t29;

    // Conditional subtraction if result >= q
    var is_gte = true;
    if t30 > 0u {
        is_gte = true;
    } else {
        for (var i: u32 = 29u; i < 30u; i = i - 1u) {
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

// ============================================================================
// BASE FIELD MONTGOMERY SQUARING (F_q) — Dedicated squaring (OPT-4)
//
// Scalar accumulators + single input = minimal register pressure.
// Generated by scripts/gen_fp_unrolled.py
// ============================================================================

fn sqr_montgomery_u384(a: U384) -> U384 {
    var t0: u32 = 0u;
    var t1: u32 = 0u;
    var t2: u32 = 0u;
    var t3: u32 = 0u;
    var t4: u32 = 0u;
    var t5: u32 = 0u;
    var t6: u32 = 0u;
    var t7: u32 = 0u;
    var t8: u32 = 0u;
    var t9: u32 = 0u;
    var t10: u32 = 0u;
    var t11: u32 = 0u;
    var t12: u32 = 0u;
    var t13: u32 = 0u;
    var t14: u32 = 0u;
    var t15: u32 = 0u;
    var t16: u32 = 0u;
    var t17: u32 = 0u;
    var t18: u32 = 0u;
    var t19: u32 = 0u;
    var t20: u32 = 0u;
    var t21: u32 = 0u;
    var t22: u32 = 0u;
    var t23: u32 = 0u;
    var t24: u32 = 0u;
    var t25: u32 = 0u;
    var t26: u32 = 0u;
    var t27: u32 = 0u;
    var t28: u32 = 0u;
    var t29: u32 = 0u;
    var t30: u32 = 0u;
    var t31: u32 = 0u;

    for (var i: u32 = 0u; i < 30u; i = i + 1u) {
        let ai = a.limbs[i];

        // Phase 1: Multiply — a[i] * a[j] (squaring)
        t0 = t0 + ai * a.limbs[0u];
        t1 = t1 + ai * a.limbs[1u];
        t2 = t2 + ai * a.limbs[2u];
        t3 = t3 + ai * a.limbs[3u];
        t4 = t4 + ai * a.limbs[4u];
        t5 = t5 + ai * a.limbs[5u];
        t6 = t6 + ai * a.limbs[6u];
        t7 = t7 + ai * a.limbs[7u];
        t8 = t8 + ai * a.limbs[8u];
        t9 = t9 + ai * a.limbs[9u];
        t10 = t10 + ai * a.limbs[10u];
        t11 = t11 + ai * a.limbs[11u];
        t12 = t12 + ai * a.limbs[12u];
        t13 = t13 + ai * a.limbs[13u];
        t14 = t14 + ai * a.limbs[14u];
        t15 = t15 + ai * a.limbs[15u];
        t16 = t16 + ai * a.limbs[16u];
        t17 = t17 + ai * a.limbs[17u];
        t18 = t18 + ai * a.limbs[18u];
        t19 = t19 + ai * a.limbs[19u];
        t20 = t20 + ai * a.limbs[20u];
        t21 = t21 + ai * a.limbs[21u];
        t22 = t22 + ai * a.limbs[22u];
        t23 = t23 + ai * a.limbs[23u];
        t24 = t24 + ai * a.limbs[24u];
        t25 = t25 + ai * a.limbs[25u];
        t26 = t26 + ai * a.limbs[26u];
        t27 = t27 + ai * a.limbs[27u];
        t28 = t28 + ai * a.limbs[28u];
        t29 = t29 + ai * a.limbs[29u];

        let m = (t0 * 0x1ffdu) & 0x1FFFu;

        // Phase 3: Reduce (literal Q constants)
        t0 = t0 + m * 0xaabu;
        t1 = t1 + m * 0x1ffdu;
        t2 = t2 + m * 0x1fffu;
        t3 = t3 + m * 0x1dffu;
        t4 = t4 + m * 0x1b9fu;
        t5 = t5 + m * 0x1fffu;
        t6 = t6 + m * 0x54fu;
        t7 = t7 + m * 0x1fd6u;
        t8 = t8 + m * 0xbffu;
        t9 = t9 + m * 0xf5u;
        t10 = t10 + m * 0x1d89u;
        t11 = t11 + m * 0xd61u;
        t12 = t12 + m * 0xa0fu;
        t13 = t13 + m * 0x1869u;
        t14 = t14 + m * 0x1d9cu;
        t15 = t15 + m * 0x257u;
        t16 = t16 + m * 0x1385u;
        t17 = t17 + m * 0x1c27u;
        t18 = t18 + m * 0x1dd2u;
        t19 = t19 + m * 0xec8u;
        t20 = t20 + m * 0x1acdu;
        t21 = t21 + m * 0x1a5u;
        t22 = t22 + m * 0x1ed9u;
        t23 = t23 + m * 0x374u;
        t24 = t24 + m * 0x1a4bu;
        t25 = t25 + m * 0x1f34u;
        t26 = t26 + m * 0xe5fu;
        t27 = t27 + m * 0x3d4u;
        t28 = t28 + m * 0x11u;
        t29 = t29 + m * 0xdu;

        // Phase 4: Carry propagation + shift (fully unrolled)
        var carry: u32 = t0 >> 13u;
        var v: u32;
        v = t1 + carry; t0 = v & 0x1FFFu; carry = v >> 13u;
        v = t2 + carry; t1 = v & 0x1FFFu; carry = v >> 13u;
        v = t3 + carry; t2 = v & 0x1FFFu; carry = v >> 13u;
        v = t4 + carry; t3 = v & 0x1FFFu; carry = v >> 13u;
        v = t5 + carry; t4 = v & 0x1FFFu; carry = v >> 13u;
        v = t6 + carry; t5 = v & 0x1FFFu; carry = v >> 13u;
        v = t7 + carry; t6 = v & 0x1FFFu; carry = v >> 13u;
        v = t8 + carry; t7 = v & 0x1FFFu; carry = v >> 13u;
        v = t9 + carry; t8 = v & 0x1FFFu; carry = v >> 13u;
        v = t10 + carry; t9 = v & 0x1FFFu; carry = v >> 13u;
        v = t11 + carry; t10 = v & 0x1FFFu; carry = v >> 13u;
        v = t12 + carry; t11 = v & 0x1FFFu; carry = v >> 13u;
        v = t13 + carry; t12 = v & 0x1FFFu; carry = v >> 13u;
        v = t14 + carry; t13 = v & 0x1FFFu; carry = v >> 13u;
        v = t15 + carry; t14 = v & 0x1FFFu; carry = v >> 13u;
        v = t16 + carry; t15 = v & 0x1FFFu; carry = v >> 13u;
        v = t17 + carry; t16 = v & 0x1FFFu; carry = v >> 13u;
        v = t18 + carry; t17 = v & 0x1FFFu; carry = v >> 13u;
        v = t19 + carry; t18 = v & 0x1FFFu; carry = v >> 13u;
        v = t20 + carry; t19 = v & 0x1FFFu; carry = v >> 13u;
        v = t21 + carry; t20 = v & 0x1FFFu; carry = v >> 13u;
        v = t22 + carry; t21 = v & 0x1FFFu; carry = v >> 13u;
        v = t23 + carry; t22 = v & 0x1FFFu; carry = v >> 13u;
        v = t24 + carry; t23 = v & 0x1FFFu; carry = v >> 13u;
        v = t25 + carry; t24 = v & 0x1FFFu; carry = v >> 13u;
        v = t26 + carry; t25 = v & 0x1FFFu; carry = v >> 13u;
        v = t27 + carry; t26 = v & 0x1FFFu; carry = v >> 13u;
        v = t28 + carry; t27 = v & 0x1FFFu; carry = v >> 13u;
        v = t29 + carry; t28 = v & 0x1FFFu; carry = v >> 13u;
        v = t30 + carry; t29 = v & 0x1FFFu; t30 = v >> 13u; t31 = 0u;
    }

    var result: U384;
    result.limbs[0u] = t0;
    result.limbs[1u] = t1;
    result.limbs[2u] = t2;
    result.limbs[3u] = t3;
    result.limbs[4u] = t4;
    result.limbs[5u] = t5;
    result.limbs[6u] = t6;
    result.limbs[7u] = t7;
    result.limbs[8u] = t8;
    result.limbs[9u] = t9;
    result.limbs[10u] = t10;
    result.limbs[11u] = t11;
    result.limbs[12u] = t12;
    result.limbs[13u] = t13;
    result.limbs[14u] = t14;
    result.limbs[15u] = t15;
    result.limbs[16u] = t16;
    result.limbs[17u] = t17;
    result.limbs[18u] = t18;
    result.limbs[19u] = t19;
    result.limbs[20u] = t20;
    result.limbs[21u] = t21;
    result.limbs[22u] = t22;
    result.limbs[23u] = t23;
    result.limbs[24u] = t24;
    result.limbs[25u] = t25;
    result.limbs[26u] = t26;
    result.limbs[27u] = t27;
    result.limbs[28u] = t28;
    result.limbs[29u] = t29;

    var is_gte = true;
    if t30 > 0u {
        is_gte = true;
    } else {
        for (var i: u32 = 29u; i < 30u; i = i - 1u) {
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
