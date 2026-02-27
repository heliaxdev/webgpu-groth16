// src/shader/bls12_381/msm.wgsl

// ============================================================================
// CONSTANTS & MONTGOMERY CONVERSION
// ============================================================================
const G1_INFINITY = PointG1(
    U384(array<u32, 12>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)),
    U384(array<u32, 12>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)),
    U384(array<u32, 12>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u))
);

const FQ2_ZERO = Fq2(
    U384(array<u32, 12>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)),
    U384(array<u32, 12>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u))
);
const G2_INFINITY = PointG2(FQ2_ZERO, FQ2_ZERO, FQ2_ZERO);

fn is_gte_q(a: U384) -> bool {
    for (var i = 11u; i < 12u; i = i - 1u) {
        if a.limbs[i] > Q_MODULUS[i] { return true; }
        if a.limbs[i] < Q_MODULUS[i] { return false; }
        if i == 0u { break; }
    }
    return true;
}

// R^2 mod q for BLS12-381 base field F_q (12 little-endian u32 limbs).
const R2_MOD_Q = U384(array<u32, 12>(
    0x1c341746u, 0xf4df1f34u, 0x09d104f1u, 0x0a76e6a6u,
    0x4c95b6d5u, 0x8de5476cu, 0x939d83c0u, 0x67eb88a9u,
    0xb519952du, 0x9a793e85u, 0x92cae3aau, 0x11988fe5u
));

// Computes a * R mod q, putting standard input into Montgomery form.
fn to_montgomery_u384(a: U384) -> U384 {
    return mul_montgomery_u384(a, R2_MOD_Q);
}

// Converts Montgomery form back to standard form
fn from_montgomery_u384(a: U384) -> U384 {
    let one = U384(array<u32, 12>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u));
    return mul_montgomery_u384(a, one);
}

fn normalize_u384(a: U384) -> U384 {
    if is_gte_q(a) {
        return sub_u384(a, U384(Q_MODULUS));
    }
    return a;
}

// Inverts a Montgomery U384 via Fermat's Little Theorem (a^(q-2) mod q)
fn invert_u384(a: U384) -> U384 {
    let q_minus_2 = array<u32, 12>(
        0xffffaaa9u, 0xb9feffffu, 0xb153ffffu, 0x1eabfffeu,
        0xf6b0f624u, 0x6730d2a0u, 0xf38512bfu, 0x64774b84u,
        0x434bacd7u, 0x4b1ba7b6u, 0x397fe69au, 0x1a0111eau
    );
    var res = to_montgomery_u384(U384(array<u32, 12>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)));
    var base = a;

    for (var i = 0u; i < 12u; i = i + 1u) {
        var limb = q_minus_2[i];
        for (var j = 0u; j < 32u; j = j + 1u) {
            if (limb & 1u) != 0u { res = mul_montgomery_u384(res, base); }
            base = mul_montgomery_u384(base, base);
            limb = limb >> 1u;
        }
    }
    return res;
}

fn to_montgomery_fp2(a: Fq2) -> Fq2 { return Fq2(to_montgomery_u384(a.c0), to_montgomery_u384(a.c1)); }
fn from_montgomery_fp2(a: Fq2) -> Fq2 { return Fq2(from_montgomery_u384(a.c0), from_montgomery_u384(a.c1)); }
fn normalize_fp2(a: Fq2) -> Fq2 { return Fq2(normalize_u384(a.c0), normalize_u384(a.c1)); }

fn invert_fp2(a: Fq2) -> Fq2 {
    let a_sq = mul_montgomery_u384(a.c0, a.c0);
    let b_sq = mul_montgomery_u384(a.c1, a.c1);
    var denom = add_u384(a_sq, b_sq);
    if is_gte_q(denom) { denom = sub_u384(denom, U384(Q_MODULUS)); }

    let inv_denom = invert_u384(denom);
    let c0 = mul_montgomery_u384(a.c0, inv_denom);
    let neg_c1 = sub_u384(U384(Q_MODULUS), a.c1);
    let c1 = mul_montgomery_u384(neg_c1, inv_denom);
    return Fq2(c0, c1);
}

// ============================================================================
// DATA LOADERS & DOMAIN BRIDGES
// ============================================================================

fn is_inf_g1(p: PointG1) -> bool {
    for (var i = 0u; i < 12u; i = i + 1u) {
        if p.z.limbs[i] != 0u { return false; }
    }
    return true;
}

fn is_inf_g2(p: PointG2) -> bool {
    for (var i = 0u; i < 12u; i = i + 1u) {
        if p.z.c0.limbs[i] != 0u || p.z.c1.limbs[i] != 0u { return false; }
    }
    return true;
}

fn add_g1_safe(p1: PointG1, p2: PointG1) -> PointG1 {
    let p1_inf = is_inf_g1(p1); let p2_inf = is_inf_g1(p2);
    if p1_inf && p2_inf { return G1_INFINITY; }
    if p1_inf { return p2; } if p2_inf { return p1; }
    return add_g1(p1, p2);
}

fn add_g2_safe(p1: PointG2, p2: PointG2) -> PointG2 {
    let p1_inf = is_inf_g2(p1); let p2_inf = is_inf_g2(p2);
    if p1_inf && p2_inf { return G2_INFINITY; }
    if p1_inf { return p2; } if p2_inf { return p1; }
    return add_g2(p1, p2);
}

// Load Standard Affine -> Montgomery Jacobian
fn load_g1(p: PointG1) -> PointG1 {
    if p.x.limbs[0] == 0u && p.y.limbs[0] == 0u && p.z.limbs[0] == 0u { return G1_INFINITY; }
    return PointG1(to_montgomery_u384(p.x), to_montgomery_u384(p.y), to_montgomery_u384(p.z));
}

// Convert Montgomery Jacobian -> Standard Affine (to return to CPU)
fn store_g1(p: PointG1) -> PointG1 {
    if is_inf_g1(p) { return G1_INFINITY; }
    let z_inv = invert_u384(p.z);
    let z_inv2 = mul_montgomery_u384(z_inv, z_inv);
    let z_inv3 = mul_montgomery_u384(z_inv2, z_inv);

    let x_aff = mul_montgomery_u384(p.x, z_inv2);
    let y_aff = mul_montgomery_u384(p.y, z_inv3);

    let z_std = U384(array<u32,12>(1u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u));
    return PointG1(
        normalize_u384(from_montgomery_u384(x_aff)),
        normalize_u384(from_montgomery_u384(y_aff)),
        z_std
    );
}

fn load_g2(p: PointG2) -> PointG2 {
    if p.x.c0.limbs[0] == 0u && p.x.c1.limbs[0] == 0u && p.z.c0.limbs[0] == 0u { return G2_INFINITY; }
    return PointG2(to_montgomery_fp2(p.x), to_montgomery_fp2(p.y), to_montgomery_fp2(p.z));
}

fn store_g2(p: PointG2) -> PointG2 {
    if is_inf_g2(p) { return G2_INFINITY; }
    let z_inv = invert_fp2(p.z);
    let z_inv2 = mul_fp2(z_inv, z_inv);
    let z_inv3 = mul_fp2(z_inv2, z_inv);

    let x_aff = mul_fp2(p.x, z_inv2);
    let y_aff = mul_fp2(p.y, z_inv3);

    let z_std = Fq2(U384(array<u32,12>(1u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u)), U384(array<u32,12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u)));
    return PointG2(
        normalize_fp2(from_montgomery_fp2(x_aff)),
        normalize_fp2(from_montgomery_fp2(y_aff)),
        z_std
    );
}

// ============================================================================
// PIPPENGER ALGORITHM PIPELINES (G1 and G2)
// ============================================================================

@group(0) @binding(0) var<storage, read> bases_g1: array<PointG1>;
@group(0) @binding(1) var<storage, read> base_indices: array<u32>;
@group(0) @binding(2) var<storage, read> bucket_pointers: array<u32>;
@group(0) @binding(3) var<storage, read> bucket_sizes: array<u32>;
@group(0) @binding(4) var<storage, read_write> aggregated_buckets_g1: array<PointG1>;

@compute @workgroup_size(64)
fn aggregate_buckets_g1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bucket_idx = global_id.x;
    if bucket_idx >= arrayLength(&bucket_pointers) { return; }

    let start = bucket_pointers[bucket_idx];
    let size = bucket_sizes[bucket_idx];
    var sum = G1_INFINITY;
    for (var i = 0u; i < size; i = i + 1u) {
        sum = add_g1_safe(sum, load_g1(bases_g1[base_indices[start + i]]));
    }
    aggregated_buckets_g1[bucket_idx] = sum;
}

@group(0) @binding(0) var<storage, read> aggregated_buckets_in_g1: array<PointG1>;
@group(0) @binding(1) var<storage, read> bucket_values: array<u32>;
@group(0) @binding(2) var<storage, read> window_starts: array<u32>;
@group(0) @binding(3) var<storage, read> window_counts: array<u32>;
@group(0) @binding(4) var<storage, read_write> window_sums_g1: array<PointG1>;

@compute @workgroup_size(1)
fn subsum_accumulation_g1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let window_id = global_id.x;
    if window_id >= arrayLength(&window_starts) { return; }

    let start = window_starts[window_id];
    let count = window_counts[window_id];
    if count == 0u {
        window_sums_g1[window_id] = G1_INFINITY;
        return;
    }

    var S = G1_INFINITY;
    var running_sum = G1_INFINITY;
    var bucket_ptr = start + count - 1u;
    var next_active_b = bucket_values[bucket_ptr];

    for (var b = next_active_b; b > 0u; b = b - 1u) {
        if b == next_active_b {
            running_sum = add_g1_safe(running_sum, aggregated_buckets_in_g1[bucket_ptr]);
            if bucket_ptr > start {
                bucket_ptr = bucket_ptr - 1u;
                next_active_b = bucket_values[bucket_ptr];
            } else {
                next_active_b = 0u;
            }
        }
        S = add_g1_safe(S, running_sum);
    }
    window_sums_g1[window_id] = store_g1(S);
}

// ==== G2 Pipelines ====

@group(0) @binding(0) var<storage, read> bases_g2: array<PointG2>;
@group(0) @binding(1) var<storage, read> base_indices_g2: array<u32>;
@group(0) @binding(2) var<storage, read> bucket_pointers_g2: array<u32>;
@group(0) @binding(3) var<storage, read> bucket_sizes_g2: array<u32>;
@group(0) @binding(4) var<storage, read_write> aggregated_buckets_g2: array<PointG2>;

@compute @workgroup_size(64)
fn aggregate_buckets_g2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bucket_idx = global_id.x;
    if bucket_idx >= arrayLength(&bucket_pointers_g2) { return; }

    let start = bucket_pointers_g2[bucket_idx];
    let size = bucket_sizes_g2[bucket_idx];
    var sum = G2_INFINITY;
    for (var i = 0u; i < size; i = i + 1u) {
        sum = add_g2_safe(sum, load_g2(bases_g2[base_indices_g2[start + i]]));
    }
    aggregated_buckets_g2[bucket_idx] = sum;
}

@group(0) @binding(0) var<storage, read> aggregated_buckets_in_g2: array<PointG2>;
@group(0) @binding(1) var<storage, read> bucket_values_g2: array<u32>;
@group(0) @binding(2) var<storage, read> window_starts_g2: array<u32>;
@group(0) @binding(3) var<storage, read> window_counts_g2: array<u32>;
@group(0) @binding(4) var<storage, read_write> window_sums_g2: array<PointG2>;

@compute @workgroup_size(1)
fn subsum_accumulation_g2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let window_id = global_id.x;
    if window_id >= arrayLength(&window_starts_g2) { return; }

    let start = window_starts_g2[window_id];
    let count = window_counts_g2[window_id];
    if count == 0u {
        window_sums_g2[window_id] = G2_INFINITY;
        return;
    }

    var S = G2_INFINITY;
    var running_sum = G2_INFINITY;
    var bucket_ptr = start + count - 1u;
    var next_active_b = bucket_values_g2[bucket_ptr];

    for (var b = next_active_b; b > 0u; b = b - 1u) {
        if b == next_active_b {
            running_sum = add_g2_safe(running_sum, aggregated_buckets_in_g2[bucket_ptr]);
            if bucket_ptr > start {
                bucket_ptr = bucket_ptr - 1u;
                next_active_b = bucket_values_g2[bucket_ptr];
            } else {
                next_active_b = 0u;
            }
        }
        S = add_g2_safe(S, running_sum);
    }
    window_sums_g2[window_id] = store_g2(S);
}

// ==== Final window reduction to a single MSM result ====

const MSM_WINDOW_BITS: u32 = 15u;

@group(0) @binding(0) var<storage, read> final_window_sums_g1: array<PointG1>;
@group(0) @binding(1) var<storage, read_write> final_result_g1: array<PointG1>;

@compute @workgroup_size(1)
fn reduce_windows_g1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x > 0u { return; }

    let n = arrayLength(&final_window_sums_g1);
    var acc = G1_INFINITY;

    for (var k = n; k > 0u; k = k - 1u) {
        if k != n {
            for (var i = 0u; i < MSM_WINDOW_BITS; i = i + 1u) {
                acc = add_g1_safe(acc, acc);
            }
        }
        acc = add_g1_safe(acc, load_g1(final_window_sums_g1[k - 1u]));
    }

    final_result_g1[0] = store_g1(acc);
}

@group(0) @binding(0) var<storage, read> final_window_sums_g2: array<PointG2>;
@group(0) @binding(1) var<storage, read_write> final_result_g2: array<PointG2>;

@compute @workgroup_size(1)
fn reduce_windows_g2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x > 0u { return; }

    let n = arrayLength(&final_window_sums_g2);
    var acc = G2_INFINITY;

    for (var k = n; k > 0u; k = k - 1u) {
        if k != n {
            for (var i = 0u; i < MSM_WINDOW_BITS; i = i + 1u) {
                acc = add_g2_safe(acc, acc);
            }
        }
        acc = add_g2_safe(acc, load_g2(final_window_sums_g2[k - 1u]));
    }

    final_result_g2[0] = store_g2(acc);
}

// ==== Debug round-trip kernels (test-only usage from Rust) ====

@group(0) @binding(0) var<storage, read> rt_in_g1: array<PointG1>;
@group(0) @binding(1) var<storage, read_write> rt_out_g1: array<PointG1>;

@compute @workgroup_size(64)
fn roundtrip_g1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&rt_in_g1) { return; }
    rt_out_g1[i] = store_g1(load_g1(rt_in_g1[i]));
}

@group(0) @binding(0) var<storage, read> rt_in_g2: array<PointG2>;
@group(0) @binding(1) var<storage, read_write> rt_out_g2: array<PointG2>;

@compute @workgroup_size(64)
fn roundtrip_g2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&rt_in_g2) { return; }
    rt_out_g2[i] = store_g2(load_g2(rt_in_g2[i]));
}

@group(0) @binding(0) var<storage, read> rt_in_coords_g1: array<PointG1>;
@group(0) @binding(1) var<storage, read_write> rt_out_coords_g1: array<PointG1>;

@compute @workgroup_size(64)
fn roundtrip_coords_g1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&rt_in_coords_g1) { return; }

    let p = rt_in_coords_g1[i];
    let x = normalize_u384(from_montgomery_u384(to_montgomery_u384(p.x)));
    let y = normalize_u384(from_montgomery_u384(to_montgomery_u384(p.y)));
    rt_out_coords_g1[i] = PointG1(x, y, p.z);
}
