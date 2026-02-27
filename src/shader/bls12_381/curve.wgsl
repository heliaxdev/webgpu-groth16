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
    // H = U2 - U1
    let h = sub_mod_q(u2, u1);
    // R = S2 - S1
    let r = sub_mod_q(s2, s1);

    // HH = H^2
    let hh = mul_montgomery_u384(h, h);
    // HHH = H * HH
    let hhh = mul_montgomery_u384(h, hh);

    // V = U1 * HH
    let v = mul_montgomery_u384(u1, hh);

    // X3 = R^2 - HHH - 2*V
    let r_sq = mul_montgomery_u384(r, r);
    var x3 = sub_mod_q(r_sq, hhh);
    x3 = sub_mod_q(x3, add_mod_q(v, v));

    // Y3 = R * (V - X3) - S1 * HHH
    let v_minus_x3 = sub_mod_q(v, x3);
    let r_times_v_minus_x3 = mul_montgomery_u384(r, v_minus_x3);
    let s1_hhh = mul_montgomery_u384(s1, hhh);
    let y3 = sub_mod_q(r_times_v_minus_x3, s1_hhh);

    // Z3 = Z1 * Z2 * H
    let z1z2 = mul_montgomery_u384(p1.z, p2.z);
    let z3 = mul_montgomery_u384(z1z2, h);

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

    // H = U2 - U1
    let h = sub_fp2(u2, u1);
    // R = S2 - S1
    let r = sub_fp2(s2, s1);

    // HH = H^2
    let hh = mul_fp2(h, h);
    // HHH = H * HH
    let hhh = mul_fp2(h, hh);

    // V = U1 * HH
    let v = mul_fp2(u1, hh);

    // X3 = R^2 - HHH - 2*V
    let r_sq = mul_fp2(r, r);
    var x3 = sub_fp2(r_sq, hhh);
    x3 = sub_fp2(x3, add_fp2(v, v));

    // Y3 = R * (V - X3) - S1 * HHH
    let v_minus_x3 = sub_fp2(v, x3);
    let r_times_v_minus_x3 = mul_fp2(r, v_minus_x3);
    let s1_hhh = mul_fp2(s1, hhh);
    let y3 = sub_fp2(r_times_v_minus_x3, s1_hhh);

    // Z3 = Z1 * Z2 * H
    let z1z2 = mul_fp2(p1.z, p2.z);
    let z3 = mul_fp2(z1z2, h);

    return PointG2(x3, y3, z3);
}
