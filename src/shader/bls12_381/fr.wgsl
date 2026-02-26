// src/shader/bls12_381/fr.wgsl

// ============================================================================
// SCALAR FIELD (F_r) - 256-bit arithmetic
// Used for NTT and polynomial evaluation.
// Limbs are stored in little-endian order (limbs[0] is least significant).
// ============================================================================

struct U256 {
    limbs: array<u32, 8>,
}

// Computes `a + b`, returning the 256-bit result (ignores overflow beyond 256 bits).
fn add_u256(a: U256, b: U256) -> U256 {
    var result: U256;
    var carry: u32 = 0u;
    
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let a_val = a.limbs[i];
        let b_val = b.limbs[i];
        
        // Manual carry tracking
        let sum1 = a_val + b_val;
        let carry1 = u32(sum1 < a_val);
        
        let sum2 = sum1 + carry;
        let carry2 = u32(sum2 < sum1);
        
        result.limbs[i] = sum2;
        carry = carry1 + carry2;
    }
    
    return result;
}

// Computes `a - b`, returning the 256-bit result.
fn sub_u256(a: U256, b: U256) -> U256 {
    var result: U256;
    var borrow: u32 = 0u;
    
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
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
