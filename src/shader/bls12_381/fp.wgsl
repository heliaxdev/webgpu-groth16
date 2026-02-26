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
const INV_Q: u32 = 0x89f3fffdu;

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
    var t = array<u32, 9>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);

    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        var carry: u32 = 0u;
        
        // Multiply step
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            let prod = mul_u32(a.limbs[i], b.limbs[j]);
            let sum1 = t[j] + carry;
            let c1 = u32(sum1 < t[j]);
            let sum2 = sum1 + prod.x;
            let c2 = u32(sum2 < sum1);
            
            t[j] = sum2;
            carry = prod.y + c1 + c2;
        }
        t[8] = t[8] + carry;

        // Reduce step
        let m = t[0] * INV_R;
        carry = 0u;
        
        // First reduction loop iteration unrolled (j=0)
        let prod0 = mul_u32(m, R_MODULUS[0]);
        let sum_m0_1 = t[0] + carry;
        let c_m0_1 = u32(sum_m0_1 < t[0]);
        let sum_m0_2 = sum_m0_1 + prod0.x;
        let c_m0_2 = u32(sum_m0_2 < sum_m0_1);
        carry = prod0.y + c_m0_1 + c_m0_2;
        
        for (var j: u32 = 1u; j < 8u; j = j + 1u) {
            let prod = mul_u32(m, R_MODULUS[j]);
            let sum1 = t[j] + carry;
            let c1 = u32(sum1 < t[j]);
            let sum2 = sum1 + prod.x;
            let c2 = u32(sum2 < sum1);
            
            t[j - 1u] = sum2;
            carry = prod.y + c1 + c2;
        }
        let sum_last = t[8] + carry;
        let c_last = u32(sum_last < t[8]);
        t[7] = sum_last;
        t[8] = c_last;
    }

    var result: U256;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        result.limbs[i] = t[i];
    }

    // Final subtraction if result >= R_MODULUS
    var is_gte = true;
    for (var i: u32 = 7u; i < 8u; i = i - 1u) {
        if (result.limbs[i] > R_MODULUS[i]) { break; }
        if (result.limbs[i] < R_MODULUS[i]) { is_gte = false; break; }
        if (i == 0u) { break; }
    }

    if (t[8] > 0u || is_gte) {
        return sub_u256(result, U256(R_MODULUS));
    }
    return result;
}

// ============================================================================
// BASE FIELD MONTGOMERY MULTIPLICATION (F_q)
// ============================================================================

fn mul_montgomery_u384(a: U384, b: U384) -> U384 {
    var t = array<u32, 13>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);

    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        var carry: u32 = 0u;
        
        for (var j: u32 = 0u; j < 12u; j = j + 1u) {
            let prod = mul_u32(a.limbs[i], b.limbs[j]);
            let sum1 = t[j] + carry;
            let c1 = u32(sum1 < t[j]);
            let sum2 = sum1 + prod.x;
            let c2 = u32(sum2 < sum1);
            
            t[j] = sum2;
            carry = prod.y + c1 + c2;
        }
        t[12] = t[12] + carry;

        let m = t[0] * INV_Q;
        carry = 0u;
        
        let prod0 = mul_u32(m, Q_MODULUS[0]);
        let sum_m0_1 = t[0] + carry;
        let c_m0_1 = u32(sum_m0_1 < t[0]);
        let sum_m0_2 = sum_m0_1 + prod0.x;
        let c_m0_2 = u32(sum_m0_2 < sum_m0_1);
        carry = prod0.y + c_m0_1 + c_m0_2;
        
        for (var j: u32 = 1u; j < 12u; j = j + 1u) {
            let prod = mul_u32(m, Q_MODULUS[j]);
            let sum1 = t[j] + carry;
            let c1 = u32(sum1 < t[j]);
            let sum2 = sum1 + prod.x;
            let c2 = u32(sum2 < sum1);
            
            t[j - 1u] = sum2;
            carry = prod.y + c1 + c2;
        }
        let sum_last = t[12] + carry;
        let c_last = u32(sum_last < t[12]);
        t[11] = sum_last;
        t[12] = c_last;
    }

    var result: U384;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        result.limbs[i] = t[i];
    }

    var is_gte = true;
    for (var i: u32 = 11u; i < 12u; i = i - 1u) {
        if (result.limbs[i] > Q_MODULUS[i]) { break; }
        if (result.limbs[i] < Q_MODULUS[i]) { is_gte = false; break; }
        if (i == 0u) { break; }
    }

    if (t[12] > 0u || is_gte) {
        return sub_u384(result, U384(Q_MODULUS));
    }
    return result;
}
