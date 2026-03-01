"""
Verify the 13-bit CIOS Montgomery multiplication and double_g1
against exact field arithmetic.
"""

q = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab

Q_LIMBS = [
    0x0aab, 0x1ffd, 0x1fff, 0x1dff, 0x1b9f, 0x1fff,
    0x054f, 0x1fd6, 0x0bff, 0x00f5, 0x1d89, 0x0d61,
    0x0a0f, 0x1869, 0x1d9c, 0x0257, 0x1385, 0x1c27,
    0x1dd2, 0x0ec8, 0x1acd, 0x01a5, 0x1ed9, 0x0374,
    0x1a4b, 0x1f34, 0x0e5f, 0x03d4, 0x0011, 0x000d,
]

INV_Q = 0x1ffd  # -q^{-1} mod 2^13

R = 1 << 390  # Montgomery R

R2_LIMBS = [
    0x070f, 0x0880, 0x10d1, 0x0c83, 0x1aec, 0x1121,
    0x004c, 0x1874, 0x066e, 0x1b75, 0x01eb, 0x1bea,
    0x07b1, 0x1f70, 0x117b, 0x0362, 0x0ed2, 0x090f,
    0x110a, 0x1482, 0x0f70, 0x1699, 0x05dc, 0x1200,
    0x0c97, 0x0c8c, 0x12b3, 0x1dc0, 0x1696, 0x0007,
]

def limbs_to_int(limbs):
    """Convert 30x13-bit limbs to integer."""
    val = 0
    for i, l in enumerate(limbs):
        val |= l << (i * 13)
    return val

def int_to_limbs(val):
    """Convert integer to 30x13-bit limbs."""
    limbs = []
    for _ in range(30):
        limbs.append(val & 0x1FFF)
        val >>= 13
    return limbs

def add_u384(a, b):
    result = [0] * 30
    carry = 0
    for i in range(30):
        s = a[i] + b[i] + carry
        result[i] = s & 0x1FFF
        carry = s >> 13
    return result

def sub_u384(a, b):
    result = [0] * 30
    borrow = 0
    for i in range(30):
        diff = a[i] + 0x2000 - b[i] - borrow
        result[i] = diff & 0x1FFF
        borrow = 1 - (diff >> 13)
    return result

def is_gte_q(a):
    for i in range(29, -1, -1):
        if a[i] > Q_LIMBS[i]:
            return True
        if a[i] < Q_LIMBS[i]:
            return False
    return True  # equal

def add_mod_q(a, b):
    s = add_u384(a, b)
    if is_gte_q(s):
        s = sub_u384(s, Q_LIMBS)
    return s

def sub_mod_q(a, b):
    is_less = False
    for i in range(29, -1, -1):
        if a[i] < b[i]:
            is_less = True
            break
        if a[i] > b[i]:
            break
    diff = sub_u384(a, b)
    if is_less:
        diff = add_u384(diff, Q_LIMBS)
    return diff

def mul_montgomery_u384(a, b):
    """13-bit lazy CIOS Montgomery multiplication."""
    t = [0] * 32

    for i in range(30):
        ai = a[i]

        # Phase 1: Multiply (lazy, no carries)
        for j in range(30):
            t[j] = t[j] + ai * b[j]

        # Phase 2: compute m
        m = (t[0] * INV_Q) & 0x1FFF

        # Phase 3: Reduce (lazy, no carries)
        for j in range(30):
            t[j] = t[j] + m * Q_LIMBS[j]

        # Phase 4: Carry propagation + shift
        carry = t[0] >> 13
        for j in range(1, 30):
            val = t[j] + carry
            t[j - 1] = val & 0x1FFF
            carry = val >> 13
        val30 = t[30] + carry
        t[29] = val30 & 0x1FFF
        t[30] = val30 >> 13
        t[31] = 0

    result = t[:30]

    # Conditional subtraction
    if t[30] > 0 or is_gte_q(result):
        result = sub_u384(result, Q_LIMBS)

    return result

def sqr_montgomery_u384(a):
    return mul_montgomery_u384(a, a)

def to_montgomery(a_limbs):
    return mul_montgomery_u384(a_limbs, R2_LIMBS)

def from_montgomery(a_limbs):
    one = [0] * 30
    one[0] = 1
    return mul_montgomery_u384(a_limbs, one)

def double_g1(px, py, pz):
    """Exact replica of the WGSL double_g1 function."""
    xx = sqr_montgomery_u384(px)
    yy = sqr_montgomery_u384(py)
    yyyy = sqr_montgomery_u384(yy)

    x_plus_yy = add_u384(px, yy)  # OPT-30: unreduced
    x_plus_yy_sq = sqr_montgomery_u384(x_plus_yy)
    s = sub_mod_q(x_plus_yy_sq, xx)
    s = sub_mod_q(s, yyyy)
    s = add_mod_q(s, s)  # 2S

    m = add_mod_q(add_mod_q(xx, xx), xx)  # 3*XX

    m_sq = sqr_montgomery_u384(m)
    t = sub_mod_q(m_sq, add_mod_q(s, s))

    x_out = t

    s_minus_t = sub_mod_q(s, t)
    m_times_s_minus_t = mul_montgomery_u384(m, s_minus_t)
    eight_yyyy = add_mod_q(yyyy, yyyy)  # 2
    eight_yyyy = add_mod_q(eight_yyyy, eight_yyyy)  # 4
    eight_yyyy = add_mod_q(eight_yyyy, eight_yyyy)  # 8
    y_out = sub_mod_q(m_times_s_minus_t, eight_yyyy)

    zz = sqr_montgomery_u384(pz)
    y_plus_z = add_u384(py, pz)  # OPT-30: unreduced
    y_plus_z_sq = sqr_montgomery_u384(y_plus_z)
    z_out = sub_mod_q(y_plus_z_sq, yy)
    z_out = sub_mod_q(z_out, zz)

    return x_out, y_out, z_out

# === Test with BLS12-381 generator ===

# Generator coordinates (standard form, big-endian hex):
# x = 0x17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb
# y = 0x08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1

x_std = 0x17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb
y_std = 0x08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1

# Verify point is on curve: y^2 = x^3 + 4
assert pow(y_std, 2, q) == (pow(x_std, 3, q) + 4) % q, "Generator not on curve!"
print("Generator on curve: ✓")

# Convert to 13-bit limbs (standard form)
x_limbs = int_to_limbs(x_std)
y_limbs = int_to_limbs(y_std)
z_limbs = int_to_limbs(1)

# Convert to Montgomery form
x_mont = to_montgomery(x_limbs)
y_mont = to_montgomery(y_limbs)
z_mont = to_montgomery(z_limbs)

# Verify Montgomery conversion
x_back = limbs_to_int(from_montgomery(x_mont)) % q
y_back = limbs_to_int(from_montgomery(y_mont)) % q
assert x_back == x_std, f"x roundtrip failed: {hex(x_back)} != {hex(x_std)}"
assert y_back == y_std, f"y roundtrip failed"
print("Montgomery roundtrip: ✓")

# Double the point using our CIOS-based implementation
x2, y2, z2 = double_g1(x_mont, y_mont, z_mont)

# Convert back to standard form
x2_std = limbs_to_int(from_montgomery(x2)) % q
y2_std = limbs_to_int(from_montgomery(y2)) % q
z2_std = limbs_to_int(from_montgomery(z2)) % q

# Convert to affine: x_aff = x/z^2, y_aff = y/z^3
z2_inv = pow(z2_std, q - 2, q)
z2_inv2 = (z2_inv * z2_inv) % q
z2_inv3 = (z2_inv2 * z2_inv) % q
x2_aff = (x2_std * z2_inv2) % q
y2_aff = (y2_std * z2_inv3) % q

print(f"\nOur double_g1 result (affine):")
print(f"  x = {hex(x2_aff)}")
print(f"  y = {hex(y2_aff)}")

# Verify on curve
rhs = (pow(x2_aff, 3, q) + 4) % q
on_curve = pow(y2_aff, 2, q) == rhs
print(f"  On curve: {on_curve}")

# Expected 2G using exact field arithmetic
# 2G formula for BLS12-381 (a=0):
# lambda = 3*x^2 / (2*y)
# x2 = lambda^2 - 2*x
# y2 = lambda*(x - x2) - y
lam_num = (3 * x_std * x_std) % q
lam_den = (2 * y_std) % q
lam_den_inv = pow(lam_den, q - 2, q)
lam = (lam_num * lam_den_inv) % q

x2_expected = (lam * lam - 2 * x_std) % q
y2_expected = (lam * (x_std - x2_expected) - y_std) % q

print(f"\nExpected 2G (affine, exact):")
print(f"  x = {hex(x2_expected)}")
print(f"  y = {hex(y2_expected)}")

# Check on curve
rhs_exp = (pow(x2_expected, 3, q) + 4) % q
print(f"  On curve: {pow(y2_expected, 2, q) == rhs_exp}")

print(f"\nMatch: x={x2_aff == x2_expected}, y={y2_aff == y2_expected}")

if x2_aff != x2_expected or y2_aff != y2_expected:
    print(f"\n!!! MISMATCH in double_g1 !!!")
    # Check intermediate values
    print(f"\n--- Intermediate values ---")
    xx = sqr_montgomery_u384(x_mont)
    print(f"xx (mont) = {hex(limbs_to_int(xx))}")
    print(f"xx (std)  = {hex(limbs_to_int(from_montgomery(xx)) % q)}")
    expected_xx = (x_std * x_std) % q
    print(f"xx (exp)  = {hex(expected_xx)}")
    print(f"xx match: {limbs_to_int(from_montgomery(xx)) % q == expected_xx}")
