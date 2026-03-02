// src/shader/bls12_381/curve_g1.wgsl

// ============================================================================
// G1 CURVE ARITHMETIC (Base Field F_q)
// ============================================================================

struct PointG1 {
    @size(128) x: U384,
    @size(128) y: U384,
    @size(128) z: U384,
}

// Computes 2 * P in Jacobian coordinates.
// BLS12-381 G1 curve: y^2 = x^3 + 4. (a = 0)
fn double_g1(p: PointG1) -> PointG1 {
    // If Z == 0, the point is at infinity.
    let is_inf = (p.z.limbs[0] == 0u); // Simplified infinity check

    // XX = X^2
    let xx = sqr_montgomery_u384(p.x);
    // YY = Y^2
    let yy = sqr_montgomery_u384(p.y);
    // YYYY = YY^2
    let yyyy = sqr_montgomery_u384(yy);

    // S = 2 * ((X + YY)^2 - XX - YYYY)
    // x_plus_yy only feeds sqr_montgomery, which handles inputs < 2q < R
    let x_plus_yy = add_u384(p.x, yy);
    let x_plus_yy_sq = sqr_montgomery_u384(x_plus_yy);
    var s = sub_mod_q(x_plus_yy_sq, xx);
    s = sub_mod_q(s, yyyy);
    s = add_mod_q(s, s); // * 2

    // M = 3 * XX (since a = 0)
    // m only feeds sqr_montgomery and mul_montgomery, which handle inputs < 3q < R
    let m = add_u384(add_u384(xx, xx), xx);

    // T = M^2 - 2*S
    let m_sq = sqr_montgomery_u384(m);
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
    let zz = sqr_montgomery_u384(p.z);
    // y_plus_z only feeds sqr_montgomery, which handles inputs < 2q < R
    let y_plus_z = add_u384(p.y, p.z);
    let y_plus_z_sq = sqr_montgomery_u384(y_plus_z);
    var z_out = sub_mod_q(y_plus_z_sq, yy);
    z_out = sub_mod_q(z_out, zz);

    return PointG1(x_out, y_out, z_out);
}

// Computes P1 + P2 in Jacobian coordinates.
fn add_g1(p1: PointG1, p2: PointG1) -> PointG1 {
    // Z1Z1 = Z1^2
    let z1z1 = sqr_montgomery_u384(p1.z);
    // Z2Z2 = Z2^2
    let z2z2 = sqr_montgomery_u384(p2.z);

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
    for (var i = 0u; i < 30u; i = i + 1u) {
        if u1.limbs[i] != u2.limbs[i] {
            u_eq = false;
            break;
        }
    }

    // Explicitly handle edge cases where the incomplete Jacobian addition formula fails
    if u_eq {
        // Check if S1 == S2 (meaning Y coordinates are equal in affine space)
        var s_eq = true;
        for (var i = 0u; i < 30u; i = i + 1u) {
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
    // two_h only feeds sqr_montgomery, which handles inputs < 2q < R
    let two_h = add_u384(h, h);
    let i = sqr_montgomery_u384(two_h);
    // J = H * I
    let j = mul_montgomery_u384(h, i);
    // r = 2*(S2 - S1)
    let s2_minus_s1 = sub_mod_q(s2, s1);
    // r feeds sqr_montgomery and mul_montgomery, which handle inputs < 2q < R
    let r = add_u384(s2_minus_s1, s2_minus_s1);
    // V = U1 * I
    let v = mul_montgomery_u384(u1, i);

    // X3 = r^2 - J - 2*V
    let r_sq = sqr_montgomery_u384(r);
    var x3 = sub_mod_q(r_sq, j);
    x3 = sub_mod_q(x3, add_mod_q(v, v));

    // Y3 = r*(V - X3) - 2*S1*J
    let v_minus_x3 = sub_mod_q(v, x3);
    let r_times_v_minus_x3 = mul_montgomery_u384(r, v_minus_x3);
    // two_s1 only feeds mul_montgomery, which handles inputs < 2q < R
    let two_s1 = add_u384(s1, s1);
    let two_s1_j = mul_montgomery_u384(two_s1, j);
    let y3 = sub_mod_q(r_times_v_minus_x3, two_s1_j);

    // Z3 = ((Z1 + Z2)^2 - Z1Z1 - Z2Z2) * H
    // z1_plus_z2 only feeds sqr_montgomery, which handles inputs < 2q < R
    let z1_plus_z2 = add_u384(p1.z, p2.z);
    let z1_plus_z2_sq = sqr_montgomery_u384(z1_plus_z2);
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
    let z1z1 = sqr_montgomery_u384(p1.z);

    // u1 = X1 (since Z2 = R: X1 * z2z2 = X1 * R * R^-1 = X1)
    // u2 = X2 * Z1Z1
    let u2 = mul_montgomery_u384(p2.x, z1z1);

    // s1 = Y1 (since Z2 = R: Y1 * Z2 * z2z2 = Y1)
    // s2 = Y2 * Z1 * Z1Z1
    let s2 = mul_montgomery_u384(mul_montgomery_u384(p2.y, p1.z), z1z1);

    // Check if U1 == U2 (doubling or cancellation)
    var u_eq = true;
    for (var i = 0u; i < 30u; i = i + 1u) {
        if p1.x.limbs[i] != u2.limbs[i] { u_eq = false; break; }
    }
    if u_eq {
        var s_eq = true;
        for (var i = 0u; i < 30u; i = i + 1u) {
            if p1.y.limbs[i] != s2.limbs[i] { s_eq = false; break; }
        }
        if s_eq { return double_g1(p1); } else { return G1_INFINITY; }
    }

    // H = U2 - U1 = U2 - X1
    let h = sub_mod_q(u2, p1.x);
    // two_h only feeds sqr_montgomery, which handles inputs < 2q < R
    let two_h = add_u384(h, h);
    let i = sqr_montgomery_u384(two_h);
    let j = mul_montgomery_u384(h, i);
    // r = 2*(S2 - S1) = 2*(S2 - Y1)
    let s2_minus_s1 = sub_mod_q(s2, p1.y);
    // r feeds sqr_montgomery and mul_montgomery, which handle inputs < 2q < R
    let r = add_u384(s2_minus_s1, s2_minus_s1);
    // V = U1 * I = X1 * I
    let v = mul_montgomery_u384(p1.x, i);

    let r_sq = sqr_montgomery_u384(r);
    var x3 = sub_mod_q(r_sq, j);
    x3 = sub_mod_q(x3, add_mod_q(v, v));

    let v_minus_x3 = sub_mod_q(v, x3);
    let r_times_v_minus_x3 = mul_montgomery_u384(r, v_minus_x3);
    // two_s1 only feeds mul_montgomery, which handles inputs < 2q < R
    let two_s1 = add_u384(p1.y, p1.y);
    let two_s1_j = mul_montgomery_u384(two_s1, j);
    let y3 = sub_mod_q(r_times_v_minus_x3, two_s1_j);

    // Z3: since Z2=R, 2*Z1*Z2 = 2*mul_montgomery(Z1,R) = 2*Z1
    // two_z1 only feeds mul_montgomery, which handles inputs < 2q < R
    let two_z1 = add_u384(p1.z, p1.z);
    let z3 = mul_montgomery_u384(two_z1, h);

    return PointG1(x3, y3, z3);
}
