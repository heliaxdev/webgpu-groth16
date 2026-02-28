#!/usr/bin/env python3
"""Compute BLS12-381 Montgomery constants for 13-bit limb representation.

With 13-bit limbs:
- F_q (384 bits): ceil(384/13) = 30 limbs, R = 2^(13*30) = 2^390
- Montgomery constant: -q^{-1} mod 2^13
- R^2 mod q for conversion to Montgomery form
"""

# BLS12-381 base field modulus q
q = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffe_b153ffff_b9feffff_ffffaaab

LIMB_BITS = 13
NUM_LIMBS = 30  # ceil(384/13) = 30 (30 * 13 = 390 bits)
LIMB_MASK = (1 << LIMB_BITS) - 1  # 0x1FFF
R = 1 << (LIMB_BITS * NUM_LIMBS)  # 2^390

def to_13bit_limbs(val, num_limbs):
    """Convert an integer to an array of 13-bit limbs (little-endian)."""
    limbs = []
    for _ in range(num_limbs):
        limbs.append(val & LIMB_MASK)
        val >>= LIMB_BITS
    assert val == 0, f"Value doesn't fit in {num_limbs} limbs: remaining {val}"
    return limbs

def format_wgsl_array(limbs, name, num_limbs):
    """Format limbs as a WGSL const array."""
    lines = [f"const {name} = array<u32, {num_limbs}>("]
    for i in range(0, len(limbs), 6):
        chunk = limbs[i:i+6]
        line = "    " + ", ".join(f"0x{v:04x}u" for v in chunk)
        if i + 6 < len(limbs):
            line += ","
        lines.append(line)
    lines.append(");")
    return "\n".join(lines)

def format_wgsl_u384(limbs, name, num_limbs):
    """Format limbs as a WGSL U384 const."""
    lines = [f"const {name} = U384(array<u32, {num_limbs}>("]
    for i in range(0, len(limbs), 6):
        chunk = limbs[i:i+6]
        line = "    " + ", ".join(f"0x{v:04x}u" for v in chunk)
        if i + 6 < len(limbs):
            line += ","
        lines.append(line)
    lines.append("));")
    return "\n".join(lines)

print(f"BLS12-381 Base Field Constants for {LIMB_BITS}-bit limbs")
print(f"=" * 60)
print(f"q = {q:#x}")
print(f"R = 2^{LIMB_BITS * NUM_LIMBS} = 2^{LIMB_BITS * NUM_LIMBS}")
print()

# 1. Q_MODULUS in 13-bit limbs
q_limbs = to_13bit_limbs(q, NUM_LIMBS)
print("// Q_MODULUS: BLS12-381 base field modulus in 30 × 13-bit limbs")
print(format_wgsl_array(q_limbs, "Q_MODULUS", NUM_LIMBS))
print()

# 2. INV_Q: -q^{-1} mod 2^13
# We need q_inv such that q * q_inv ≡ -1 (mod 2^13)
# Equivalently: q_inv ≡ (-q)^{-1} (mod 2^13) => we want -(q^{-1}) mod 2^13
q_mod = q % (1 << LIMB_BITS)
q_inv = pow(q_mod, -1, 1 << LIMB_BITS)
inv_q = (-(q_inv)) % (1 << LIMB_BITS)
# Verify: q * inv_q ≡ -1 (mod 2^13)
assert (q * inv_q + 1) % (1 << LIMB_BITS) == 0, "INV_Q verification failed"
print(f"// INV_Q: -q^{{-1}} mod 2^{LIMB_BITS}")
print(f"const INV_Q: u32 = 0x{inv_q:04x}u;")
print()

# 3. R^2 mod q for Montgomery conversion
r2_mod_q = pow(R, 2, q)
r2_limbs = to_13bit_limbs(r2_mod_q, NUM_LIMBS)
print("// R2_MOD_Q: R^2 mod q for Montgomery conversion (R = 2^390)")
print(format_wgsl_u384(r2_limbs, "R2_MOD_Q", NUM_LIMBS))
print()

# 4. q - 2 for Fermat inversion (a^(q-2) mod q)
q_minus_2 = q - 2
qm2_limbs = to_13bit_limbs(q_minus_2, NUM_LIMBS)
print("// Q_MINUS_2: q - 2 for Fermat's little theorem inversion")
print(format_wgsl_array(qm2_limbs, "Q_MINUS_2", NUM_LIMBS))
print()

# 5. Montgomery representation of 1: R mod q
mont_one = R % q
mont_one_limbs = to_13bit_limbs(mont_one, NUM_LIMBS)
print("// MONT_ONE: R mod q (Montgomery representation of 1)")
print(format_wgsl_u384(mont_one_limbs, "MONT_ONE", NUM_LIMBS))
print()

# Verify: MONT_ONE * MONT_ONE * R^{-1} mod q = MONT_ONE (since 1*1 = 1 in Montgomery)
r_inv = pow(R, -1, q)
mont_mul_result = (mont_one * mont_one * r_inv) % q
assert mont_mul_result == mont_one, "MONT_ONE self-multiplication check failed"
print("// Verification: MONT_ONE * MONT_ONE * R^{-1} mod q = MONT_ONE ✓")
print()

# 6. Verify the existing 32-bit constants match q
print("// Cross-check: q in 12 × 32-bit limbs (should match existing code)")
q_32 = to_13bit_limbs(q, NUM_LIMBS)  # already done above
q_32bit = []
val = q
for _ in range(12):
    q_32bit.append(val & 0xFFFFFFFF)
    val >>= 32
print("// " + ", ".join(f"0x{v:08x}" for v in q_32bit))
print()

# 7. Also compute INV_Q for 32-bit (verify existing)
q_mod32 = q % (1 << 32)
q_inv32 = pow(q_mod32, -1, 1 << 32)
inv_q32 = (-(q_inv32)) % (1 << 32)
print(f"// Existing INV_Q (32-bit): 0x{inv_q32:08x}  (should be 0xfffcfffd)")
print()

# 8. Summary
print("=" * 60)
print("SUMMARY")
print(f"  Limb width: {LIMB_BITS} bits")
print(f"  Number of limbs: {NUM_LIMBS}")
print(f"  R = 2^{LIMB_BITS * NUM_LIMBS}")
print(f"  INV_Q = 0x{inv_q:04x}")
print(f"  Max limb value: 0x{LIMB_MASK:04x} ({LIMB_MASK})")
print(f"  Max product: {LIMB_MASK**2} ({LIMB_MASK**2:#x}) fits in u32: {LIMB_MASK**2 < 2**32}")
print(f"  Max accumulation (30 products): {30 * LIMB_MASK**2} ({30 * LIMB_MASK**2:#x}) fits in u32: {30 * LIMB_MASK**2 < 2**32}")
