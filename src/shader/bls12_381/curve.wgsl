// src/shader/bls12_381/curve.wgsl

// ============================================================================
// G1 CURVE ARITHMETIC (Base Field F_q)
// ============================================================================

struct PointG1 {
    x: U384,
    y: U384,
    z: U384,
}

// Computes 2 * P in Jacobian coordinates.
// BLS12-381 G1 curve: y^2 = x^3 + 4. (a = 0)
fn double_g1(p: PointG1) -> PointG1 {
    // If Z == 0, the point is at infinity.
    let is_inf = (p.z.limbs[0] == 0u); // Simplified infinity check

    // XX = X^2
    let xx = mul_montgomery_u384(p.x, p.x);
    // YY = Y^2
    let yy = mul_montgomery_u384(p.y, p.y);
    // YYYY = YY^2
    let yyyy = mul_montgomery_u384(yy, yy);
    
    // S = 2 * ((X + YY)^2 - XX - YYYY)
    let x_plus_yy = add_mod_q(p.x, yy);
    let x_plus_yy_sq = mul_montgomery_u384(x_plus_yy, x_plus_yy);
    var s = sub_mod_q(x_plus_yy_sq, xx);
    s = sub_mod_q(s, yyyy);
    s = add_mod_q(s, s); // * 2

    // M = 3 * XX (since a = 0)
    let m = add_mod_q(add_mod_q(xx, xx), xx);

    // T = M^2 - 2*S
    let m_sq = mul_montgomery_u384(m, m);
    let t = sub_mod_q(m_sq, add_mod_q(s, s));

    // X_out = T
    let x_out = t;

    // Y_out = M * (S - T) - 8 * YYYY
    let s_minus_t = sub_mod_q(s, t);
    let m_times_s_minus_t = mul_montgomery_u384(m, s_minus_t);
    var eight_yyyy = add_mod_q(yyyy, yyyy); // 2
    eight_yyyy = add_mod_q(eight_yyyy, eight_yyyy); // 4
    eight_yyyy = add_mod_q(eight_yyyy, eight_yyyy); // 8
    let y_out = sub_mod_q(m_times_s_minus_t, eight_yyyy);

    // Z_out = (Y + Z)^2 - YY - ZZ
    let zz = mul_montgomery_u384(p.z, p.z);
    let y_plus_z = add_mod_q(p.y, p.z);
    let y_plus_z_sq = mul_montgomery_u384(y_plus_z, y_plus_z);
    var z_out = sub_mod_q(y_plus_z_sq, yy);
    z_out = sub_mod_q(z_out, zz);

    return PointG1(x_out, y_out, z_out);
}

// Computes P1 + P2 in Jacobian coordinates.
fn add_g1(p1: PointG1, p2: PointG1) -> PointG1 {
    // Z1Z1 = Z1^2
    let z1z1 = mul_montgomery_u384(p1.z, p1.z);
    // Z2Z2 = Z2^2
    let z2z2 = mul_montgomery_u384(p2.z, p2.z);

    // U1 = X1 * Z2Z2
    let u1 = mul_montgomery_u384(p1.x, z2z2);
    // U2 = X2 * Z1Z1
    let u2 = mul_montgomery_u384(p2.x, z1z1);

    // S1 = Y1 * Z2 * Z2Z2
    let s1 = mul_montgomery_u384(mul_montgomery_u384(p1.y, p2.z), z2z2);
    // S2 = Y2 * Z1 * Z1Z1
    let s2 = mul_montgomery_u384(mul_montgomery_u384(p2.y, p1.z), z1z1);

    // Check if U1 == U2 (meaning X coordinates are equal in affine space)
    var u_eq = true;
    for (var i = 0u; i < 12u; i = i + 1u) {
        if u1.limbs[i] != u2.limbs[i] {
            u_eq = false;
            break;
        }
    }

    // Explicitly handle edge cases where the incomplete Jacobian addition formula fails
    if u_eq {
        // Check if S1 == S2 (meaning Y coordinates are equal in affine space)
        var s_eq = true;
        for (var i = 0u; i < 12u; i = i + 1u) {
            if s1.limbs[i] != s2.limbs[i] {
                s_eq = false;
                break;
            }
        }
        if s_eq {
            // Point doubling: P1 == P2
            return double_g1(p1);
        } else {
            // Point cancellation: P1 == -P2
            return G1_INFINITY;
        }
    }

    // Proceed with standard Jacobian addition
    // Jacobian add-2007-bl formula (EFD)
    // H = U2 - U1
    let h = sub_mod_q(u2, u1);
    // I = (2*H)^2
    let two_h = add_mod_q(h, h);
    let i = mul_montgomery_u384(two_h, two_h);
    // J = H * I
    let j = mul_montgomery_u384(h, i);
    // r = 2*(S2 - S1)
    let s2_minus_s1 = sub_mod_q(s2, s1);
    let r = add_mod_q(s2_minus_s1, s2_minus_s1);
    // V = U1 * I
    let v = mul_montgomery_u384(u1, i);

    // X3 = r^2 - J - 2*V
    let r_sq = mul_montgomery_u384(r, r);
    var x3 = sub_mod_q(r_sq, j);
    x3 = sub_mod_q(x3, add_mod_q(v, v));

    // Y3 = r*(V - X3) - 2*S1*J
    let v_minus_x3 = sub_mod_q(v, x3);
    let r_times_v_minus_x3 = mul_montgomery_u384(r, v_minus_x3);
    let two_s1 = add_mod_q(s1, s1);
    let two_s1_j = mul_montgomery_u384(two_s1, j);
    let y3 = sub_mod_q(r_times_v_minus_x3, two_s1_j);

    // Z3 = ((Z1 + Z2)^2 - Z1Z1 - Z2Z2) * H
    let z1_plus_z2 = add_mod_q(p1.z, p2.z);
    let z1_plus_z2_sq = mul_montgomery_u384(z1_plus_z2, z1_plus_z2);
    let z1z2_factor = sub_mod_q(sub_mod_q(z1_plus_z2_sq, z1z1), z2z2);
    let z3 = mul_montgomery_u384(z1z2_factor, h);

    return PointG1(x3, y3, z3);
}

// Mixed addition: P1 (projective, arbitrary Z) + P2 (affine in Montgomery form, Z2 = R).
// When Z2 = R (the Montgomery representation of 1), several terms simplify:
//   z2z2 = R, u1 = X1, s1 = Y1, and z1z2_factor = 2*Z1.
// This saves 5 Montgomery multiplications compared to the full Jacobian addition
// (11 muls vs 16 muls).
fn add_g1_mixed(p1: PointG1, p2: PointG1) -> PointG1 {
    // Z1Z1 = Z1^2
    let z1z1 = mul_montgomery_u384(p1.z, p1.z);

    // u1 = X1 (since Z2 = R: X1 * z2z2 = X1 * R * R^-1 = X1)
    // u2 = X2 * Z1Z1
    let u2 = mul_montgomery_u384(p2.x, z1z1);

    // s1 = Y1 (since Z2 = R: Y1 * Z2 * z2z2 = Y1)
    // s2 = Y2 * Z1 * Z1Z1
    let s2 = mul_montgomery_u384(mul_montgomery_u384(p2.y, p1.z), z1z1);

    // Check if U1 == U2 (doubling or cancellation)
    var u_eq = true;
    for (var i = 0u; i < 12u; i = i + 1u) {
        if p1.x.limbs[i] != u2.limbs[i] { u_eq = false; break; }
    }
    if u_eq {
        var s_eq = true;
        for (var i = 0u; i < 12u; i = i + 1u) {
            if p1.y.limbs[i] != s2.limbs[i] { s_eq = false; break; }
        }
        if s_eq { return double_g1(p1); }
        else { return G1_INFINITY; }
    }

    // H = U2 - U1 = U2 - X1
    let h = sub_mod_q(u2, p1.x);
    let two_h = add_mod_q(h, h);
    let i = mul_montgomery_u384(two_h, two_h);
    let j = mul_montgomery_u384(h, i);
    // r = 2*(S2 - S1) = 2*(S2 - Y1)
    let s2_minus_s1 = sub_mod_q(s2, p1.y);
    let r = add_mod_q(s2_minus_s1, s2_minus_s1);
    // V = U1 * I = X1 * I
    let v = mul_montgomery_u384(p1.x, i);

    let r_sq = mul_montgomery_u384(r, r);
    var x3 = sub_mod_q(r_sq, j);
    x3 = sub_mod_q(x3, add_mod_q(v, v));

    let v_minus_x3 = sub_mod_q(v, x3);
    let r_times_v_minus_x3 = mul_montgomery_u384(r, v_minus_x3);
    let two_s1 = add_mod_q(p1.y, p1.y);
    let two_s1_j = mul_montgomery_u384(two_s1, j);
    let y3 = sub_mod_q(r_times_v_minus_x3, two_s1_j);

    // Z3: since Z2=R, 2*Z1*Z2 = 2*mul_montgomery(Z1,R) = 2*Z1
    let two_z1 = add_mod_q(p1.z, p1.z);
    let z3 = mul_montgomery_u384(two_z1, h);

    return PointG1(x3, y3, z3);
}

// ============================================================================
// EXTENSION FIELD ARITHMETIC (F_q^2)
// Required for G2 Curve elements (the 'B' element in Groth16)
// BLS12-381 F_q^2 is defined as F_q[u] / (u^2 + 1)
// ============================================================================

struct Fq2 {
    c0: U384,
    c1: U384,
}

fn add_fp2(a: Fq2, b: Fq2) -> Fq2 {
    return Fq2(add_mod_q(a.c0, b.c0), add_mod_q(a.c1, b.c1));
}

fn sub_fp2(a: Fq2, b: Fq2) -> Fq2 {
    return Fq2(sub_mod_q(a.c0, b.c0), sub_mod_q(a.c1, b.c1));
}

// (a0 + a1*u) * (b0 + b1*u) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*u
fn mul_fp2(a: Fq2, b: Fq2) -> Fq2 {
    let a0b0 = mul_montgomery_u384(a.c0, b.c0);
    let a1b1 = mul_montgomery_u384(a.c1, b.c1);
    
    // a0b1 + a1b0 = (a0 + a1)*(b0 + b1) - a0b0 - a1b1
    let a0_plus_a1 = add_mod_q(a.c0, a.c1);
    let b0_plus_b1 = add_mod_q(b.c0, b.c1);
    let a_plus_b_prod = mul_montgomery_u384(a0_plus_a1, b0_plus_b1);

    var c1_out = sub_mod_q(a_plus_b_prod, a0b0);
    c1_out = sub_mod_q(c1_out, a1b1);

    let c0_out = sub_mod_q(a0b0, a1b1);

    return Fq2(c0_out, c1_out);
}

// ============================================================================
// G2 CURVE ARITHMETIC (Extension Field F_q^2)
// ============================================================================

struct PointG2 {
    x: Fq2,
    y: Fq2,
    z: Fq2,
}

// Computes 2 * P in Jacobian coordinates for G2.
// BLS12-381 G2 curve: y^2 = x^3 + 4(1+i). (a = 0)
fn double_g2(p: PointG2) -> PointG2 {
    // XX = X^2
    let xx = mul_fp2(p.x, p.x);
    // YY = Y^2
    let yy = mul_fp2(p.y, p.y);
    // YYYY = YY^2
    let yyyy = mul_fp2(yy, yy);

    // S = 2 * ((X + YY)^2 - XX - YYYY)
    let x_plus_yy = add_fp2(p.x, yy);
    let x_plus_yy_sq = mul_fp2(x_plus_yy, x_plus_yy);
    var s = sub_fp2(x_plus_yy_sq, xx);
    s = sub_fp2(s, yyyy);
    s = add_fp2(s, s); // * 2

    // M = 3 * XX (since a = 0)
    let m = add_fp2(add_fp2(xx, xx), xx);

    // T = M^2 - 2*S
    let m_sq = mul_fp2(m, m);
    let t = sub_fp2(m_sq, add_fp2(s, s));

    // X_out = T
    let x_out = t;

    // Y_out = M * (S - T) - 8 * YYYY
    let s_minus_t = sub_fp2(s, t);
    let m_times_s_minus_t = mul_fp2(m, s_minus_t);
    var eight_yyyy = add_fp2(yyyy, yyyy); // 2
    eight_yyyy = add_fp2(eight_yyyy, eight_yyyy); // 4
    eight_yyyy = add_fp2(eight_yyyy, eight_yyyy); // 8
    let y_out = sub_fp2(m_times_s_minus_t, eight_yyyy);

    // Z_out = (Y + Z)^2 - YY - ZZ
    let zz = mul_fp2(p.z, p.z);
    let y_plus_z = add_fp2(p.y, p.z);
    let y_plus_z_sq = mul_fp2(y_plus_z, y_plus_z);
    var z_out = sub_fp2(y_plus_z_sq, yy);
    z_out = sub_fp2(z_out, zz);

    return PointG2(x_out, y_out, z_out);
}

// Computes P1 + P2 in Jacobian coordinates for G2.
fn add_g2(p1: PointG2, p2: PointG2) -> PointG2 {
    // Z1Z1 = Z1^2
    let z1z1 = mul_fp2(p1.z, p1.z);
    // Z2Z2 = Z2^2
    let z2z2 = mul_fp2(p2.z, p2.z);

    // U1 = X1 * Z2Z2
    let u1 = mul_fp2(p1.x, z2z2);
    // U2 = X2 * Z1Z1
    let u2 = mul_fp2(p2.x, z1z1);

    // S1 = Y1 * Z2 * Z2Z2
    let s1 = mul_fp2(mul_fp2(p1.y, p2.z), z2z2);
    // S2 = Y2 * Z1 * Z1Z1
    let s2 = mul_fp2(mul_fp2(p2.y, p1.z), z1z1);

    // Explicitly handle edge cases where incomplete Jacobian addition fails.
    var u_eq = true;
    for (var i = 0u; i < 12u; i = i + 1u) {
        if u1.c0.limbs[i] != u2.c0.limbs[i] || u1.c1.limbs[i] != u2.c1.limbs[i] {
            u_eq = false;
            break;
        }
    }

    if u_eq {
        var s_eq = true;
        for (var i = 0u; i < 12u; i = i + 1u) {
            if s1.c0.limbs[i] != s2.c0.limbs[i] || s1.c1.limbs[i] != s2.c1.limbs[i] {
                s_eq = false;
                break;
            }
        }
        if s_eq {
            return double_g2(p1);
        } else {
            return PointG2(Fq2(U384(array<u32,12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u)), U384(array<u32,12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u))), Fq2(U384(array<u32,12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u)), U384(array<u32,12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u))), Fq2(U384(array<u32,12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u)), U384(array<u32,12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u))));
        }
    }

    // Jacobian add-2007-bl formula (EFD), lifted to Fp2.
    let h = sub_fp2(u2, u1);
    let two_h = add_fp2(h, h);
    let i = mul_fp2(two_h, two_h);
    let j = mul_fp2(h, i);
    let s2_minus_s1 = sub_fp2(s2, s1);
    let r = add_fp2(s2_minus_s1, s2_minus_s1);
    let v = mul_fp2(u1, i);

    let r_sq = mul_fp2(r, r);
    var x3 = sub_fp2(r_sq, j);
    x3 = sub_fp2(x3, add_fp2(v, v));

    let v_minus_x3 = sub_fp2(v, x3);
    let r_times_v_minus_x3 = mul_fp2(r, v_minus_x3);
    let two_s1 = add_fp2(s1, s1);
    let two_s1_j = mul_fp2(two_s1, j);
    let y3 = sub_fp2(r_times_v_minus_x3, two_s1_j);

    let z1_plus_z2 = add_fp2(p1.z, p2.z);
    let z1_plus_z2_sq = mul_fp2(z1_plus_z2, z1_plus_z2);
    let z1z2_factor = sub_fp2(sub_fp2(z1_plus_z2_sq, z1z1), z2z2);
    let z3 = mul_fp2(z1z2_factor, h);

    return PointG2(x3, y3, z3);
}

// Mixed addition: P1 (projective, arbitrary Z) + P2 (affine in Montgomery form, Z2 = (R,0)).
// When Z2 = (R,0) (Montgomery representation of (1,0) in Fq2), several terms simplify:
//   z2z2 = (R,0), u1 = X1, s1 = Y1, and z1z2_factor = 2*Z1.
// This saves 5 Fq2 multiplications compared to the full Jacobian addition
// (11 mul_fp2 vs 16 mul_fp2).
fn add_g2_mixed(p1: PointG2, p2: PointG2) -> PointG2 {
    // Z1Z1 = Z1^2
    let z1z1 = mul_fp2(p1.z, p1.z);

    // u1 = X1 (since Z2 = (R,0): X1 * z2z2 = X1)
    // u2 = X2 * Z1Z1
    let u2 = mul_fp2(p2.x, z1z1);

    // s1 = Y1 (since Z2 = (R,0): Y1 * Z2 * z2z2 = Y1)
    // s2 = Y2 * Z1 * Z1Z1
    let s2 = mul_fp2(mul_fp2(p2.y, p1.z), z1z1);

    // Check if U1 == U2 (doubling or cancellation)
    var u_eq = true;
    for (var i = 0u; i < 12u; i = i + 1u) {
        if p1.x.c0.limbs[i] != u2.c0.limbs[i] || p1.x.c1.limbs[i] != u2.c1.limbs[i] {
            u_eq = false;
            break;
        }
    }
    if u_eq {
        var s_eq = true;
        for (var i = 0u; i < 12u; i = i + 1u) {
            if p1.y.c0.limbs[i] != s2.c0.limbs[i] || p1.y.c1.limbs[i] != s2.c1.limbs[i] {
                s_eq = false;
                break;
            }
        }
        if s_eq { return double_g2(p1); }
        else {
            return PointG2(
                Fq2(U384(array<u32,12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u)),
                    U384(array<u32,12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u))),
                Fq2(U384(array<u32,12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u)),
                    U384(array<u32,12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u))),
                Fq2(U384(array<u32,12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u)),
                    U384(array<u32,12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u)))
            );
        }
    }

    // H = U2 - U1 = U2 - X1
    let h = sub_fp2(u2, p1.x);
    let two_h = add_fp2(h, h);
    let i_val = mul_fp2(two_h, two_h);
    let j = mul_fp2(h, i_val);
    // r = 2*(S2 - S1) = 2*(S2 - Y1)
    let s2_minus_s1 = sub_fp2(s2, p1.y);
    let r = add_fp2(s2_minus_s1, s2_minus_s1);
    // V = U1 * I = X1 * I
    let v = mul_fp2(p1.x, i_val);

    let r_sq = mul_fp2(r, r);
    var x3 = sub_fp2(r_sq, j);
    x3 = sub_fp2(x3, add_fp2(v, v));

    let v_minus_x3 = sub_fp2(v, x3);
    let r_times_v_minus_x3 = mul_fp2(r, v_minus_x3);
    let two_s1 = add_fp2(p1.y, p1.y);
    let two_s1_j = mul_fp2(two_s1, j);
    let y3 = sub_fp2(r_times_v_minus_x3, two_s1_j);

    // Z3: since Z2=(R,0), 2*Z1*Z2 = 2*Z1 in Montgomery
    let two_z1 = add_fp2(p1.z, p1.z);
    let z3 = mul_fp2(two_z1, h);

    return PointG2(x3, y3, z3);
}
