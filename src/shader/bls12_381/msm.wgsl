// src/shader/bls12_381/msm.wgsl

// ============================================================================
// CONSTANTS & SAFE ARITHMETIC
// ============================================================================

// The Point at Infinity in Jacobian Coordinates (Z = 0)
const G1_INFINITY = PointG1(
    U384(array<u32, 12>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)),
    U384(array<u32, 12>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)),
    U384(array<u32, 12>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u))
);

fn is_inf_g1(p: PointG1) -> bool {
    var is_zero = true;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        if p.z.limbs[i] != 0u {
            is_zero = false;
            break;
        }
    }
    return is_zero;
}

// Safe addition that gracefully handles the point at infinity
fn add_g1_safe(p1: PointG1, p2: PointG1) -> PointG1 {
    let p1_inf = is_inf_g1(p1);
    let p2_inf = is_inf_g1(p2);

    if p1_inf && p2_inf { return G1_INFINITY; }
    if p1_inf { return p2; }
    if p2_inf { return p1; }

    return add_g1(p1, p2);
}

// ============================================================================
// GPU PIPELINE BUFFERS
// ============================================================================

@group(0) @binding(0)
var<storage, read> buckets: array<PointG1>;

@group(0) @binding(1)
var<storage, read> bucket_indices: array<u32>; // The active b_i values

@group(0) @binding(2)
var<storage, read> bucket_count: u32; // The number of active buckets 'm'

@group(0) @binding(3)
var<storage, read_write> final_result: array<PointG1>;

// ============================================================================
// SUBSUM ACCUMULATION (Luo, Fu, Gong - Algorithm 3)
// ============================================================================

// Evaluates S = b_1*S_1 + b_2*S_2 + ... + b_m*S_m
// Optimized for inconsecutive bucket indices where the max difference d=6.
@compute @workgroup_size(1)
fn subsum_accumulation_g1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x != 0u { return; } // Executed by a single thread post-reduction

    let m = bucket_count;
    if m == 0u {
        final_result[0] = G1_INFINITY;
        return;
    }

    // d = 6 for BLS12-381 optimal radixes. tmp array needs length d + 1 = 7.
    var tmp = array<PointG1, 7>(
        G1_INFINITY, G1_INFINITY, G1_INFINITY, G1_INFINITY, 
        G1_INFINITY, G1_INFINITY, G1_INFINITY
    );

    var prev_b: u32 = 0u;
    if m > 0u {
        prev_b = bucket_indices[m - 1u];
    }

    // for i = m to 1 do
    for (var i: u32 = m; i > 0u; i = i - 1u) {
        let current_bucket = buckets[i - 1u];

        // tmp[0] = tmp[0] + S_i
        tmp[0] = add_g1_safe(tmp[0], current_bucket);

        let current_b = bucket_indices[i - 1u];
        var next_b: u32 = 0u;
        if i > 1u {
            next_b = bucket_indices[i - 2u];
        }

        // k = b_i - b_{i-1}
        let k = current_b - next_b;

        // if k >= 1 then tmp[k] = tmp[k] + tmp[0]
        if k >= 1u && k <= 6u {
            tmp[k] = add_g1_safe(tmp[k], tmp[0]);
        }
    }

    // Return 1*tmp[1] + 2*tmp[2] + ... + 6*tmp[6]
    var S = G1_INFINITY;
    
    // Evaluate sequentially to avoid multiplication overheads in G1
    // Using simple repeated additions for small coefficients (1 to 6)
    
    // tmp[6] * 6
    var t6 = double_g1(tmp[6]);
    t6 = add_g1_safe(t6, tmp[6]); // * 3
    t6 = double_g1(t6);           // * 6
    S = add_g1_safe(S, t6);
    
    // tmp[5] * 5
    var t5 = double_g1(tmp[5]);
    t5 = double_g1(t5);           // * 4
    t5 = add_g1_safe(t5, tmp[5]); // * 5
    S = add_g1_safe(S, t5);
    
    // tmp[4] * 4
    var t4 = double_g1(tmp[4]);
    t4 = double_g1(t4);
    S = add_g1_safe(S, t4);
    
    // tmp[3] * 3
    var t3 = double_g1(tmp[3]);
    t3 = add_g1_safe(t3, tmp[3]);
    S = add_g1_safe(S, t3);
    
    // tmp[2] * 2
    let t2 = double_g1(tmp[2]);
    S = add_g1_safe(S, t2);
    
    // tmp[1] * 1
    S = add_g1_safe(S, tmp[1]);

    final_result[0] = S;
}

// ============================================================================
// G2 CONSTANTS & SAFE ARITHMETIC
// ============================================================================

const FQ2_ZERO = Fq2(
    U384(array<u32, 12>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)),
    U384(array<u32, 12>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u))
);

// The Point at Infinity in Jacobian Coordinates for G2 (Z = 0)
const G2_INFINITY = PointG2(FQ2_ZERO, FQ2_ZERO, FQ2_ZERO);

fn is_inf_g2(p: PointG2) -> bool {
    var is_zero = true;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        if p.z.c0.limbs[i] != 0u || p.z.c1.limbs[i] != 0u {
            is_zero = false;
            break;
        }
    }
    return is_zero;
}

// Safe addition that gracefully handles the point at infinity for G2
fn add_g2_safe(p1: PointG2, p2: PointG2) -> PointG2 {
    let p1_inf = is_inf_g2(p1);
    let p2_inf = is_inf_g2(p2);

    if p1_inf && p2_inf { return G2_INFINITY; }
    if p1_inf { return p2; }
    if p2_inf { return p1; }

    return add_g2(p1, p2);
}

// ============================================================================
// G2 GPU PIPELINE BUFFERS
// ============================================================================

@group(1) @binding(0) // Used Group 1 to avoid overlap with G1, adjust to preference
var<storage, read> buckets_g2: array<PointG2>;

@group(1) @binding(1)
var<storage, read> bucket_indices_g2: array<u32>; // The active b_i values

@group(1) @binding(2)
var<storage, read> bucket_count_g2: u32; // The number of active buckets 'm'

@group(1) @binding(3)
var<storage, read_write> final_result_g2: array<PointG2>;

// ============================================================================
// G2 SUBSUM ACCUMULATION (Luo, Fu, Gong - Algorithm 3)
// ============================================================================

// Evaluates S = b_1*S_1 + b_2*S_2 + ... + b_m*S_m for G2 points
@compute @workgroup_size(1)
fn subsum_accumulation_g2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x != 0u { return; } // Executed by a single thread post-reduction

    let m = bucket_count_g2;
    if m == 0u {
        final_result_g2[0] = G2_INFINITY;
        return;
    }

    // d = 6 for BLS12-381 optimal radixes. tmp array needs length d + 1 = 7.
    var tmp = array<PointG2, 7>(
        G2_INFINITY, G2_INFINITY, G2_INFINITY, G2_INFINITY, 
        G2_INFINITY, G2_INFINITY, G2_INFINITY
    );

    var prev_b: u32 = 0u;
    if m > 0u {
        prev_b = bucket_indices_g2[m - 1u];
    }

    // for i = m to 1 do
    for (var i: u32 = m; i > 0u; i = i - 1u) {
        let current_bucket = buckets_g2[i - 1u];

        // tmp[0] = tmp[0] + S_i
        tmp[0] = add_g2_safe(tmp[0], current_bucket);

        let current_b = bucket_indices_g2[i - 1u];
        var next_b: u32 = 0u;
        if i > 1u {
            next_b = bucket_indices_g2[i - 2u];
        }

        // k = b_i - b_{i-1}
        let k = current_b - next_b;

        // if k >= 1 then tmp[k] = tmp[k] + tmp[0]
        if k >= 1u && k <= 6u {
            tmp[k] = add_g2_safe(tmp[k], tmp[0]);
        }
    }

    // Return 1*tmp[1] + 2*tmp[2] + ... + 6*tmp[6]
    var S = G2_INFINITY;

    // Evaluate sequentially to avoid multiplication overheads in G2 F_q^2
    // Using simple repeated additions for small coefficients (1 to 6)

    // tmp[6] * 6
    var t6 = double_g2(tmp[6]);
    t6 = add_g2_safe(t6, tmp[6]); // * 3
    t6 = double_g2(t6);           // * 6
    S = add_g2_safe(S, t6);

    // tmp[5] * 5
    var t5 = double_g2(tmp[5]);
    t5 = double_g2(t5);           // * 4
    t5 = add_g2_safe(t5, tmp[5]); // * 5
    S = add_g2_safe(S, t5);

    // tmp[4] * 4
    var t4 = double_g2(tmp[4]);
    t4 = double_g2(t4);
    S = add_g2_safe(S, t4);

    // tmp[3] * 3
    var t3 = double_g2(tmp[3]);
    t3 = add_g2_safe(t3, tmp[3]);
    S = add_g2_safe(S, t3);

    // tmp[2] * 2
    let t2 = double_g2(tmp[2]);
    S = add_g2_safe(S, t2);

    // tmp[1] * 1
    S = add_g2_safe(S, tmp[1]);

    final_result_g2[0] = S;
}
