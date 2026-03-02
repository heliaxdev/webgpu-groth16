// src/shader/bls12_381/msm_g2_subsum.wgsl

const U384_ZERO = U384(array<u32, 30>(
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
));

const FQ2_ZERO = Fq2(U384_ZERO, U384_ZERO);
const G2_INFINITY = PointG2(FQ2_ZERO, FQ2_ZERO, FQ2_ZERO);
const FQ2_ONE_MONT = Fq2(MONT_ONE, U384_ZERO);
const G2_PROJ_IDENTITY = PointG2(FQ2_ZERO, FQ2_ONE_MONT, FQ2_ZERO);

const R2_MOD_Q = U384(array<u32, 30>(
    0x070fu, 0x0880u, 0x10d1u, 0x0c83u, 0x1aecu, 0x1121u,
    0x004cu, 0x1874u, 0x066eu, 0x1b75u, 0x01ebu, 0x1beau,
    0x07b1u, 0x1f70u, 0x117bu, 0x0362u, 0x0ed2u, 0x090fu,
    0x110au, 0x1482u, 0x0f70u, 0x1699u, 0x05dcu, 0x1200u,
    0x0c97u, 0x0c8cu, 0x12b3u, 0x1dc0u, 0x1696u, 0x0007u
));

const MONT_ONE = U384(array<u32, 30>(
    0x1f2eu, 0x068fu, 0x0000u, 0x0c00u, 0x0467u, 0x0056u,
    0x0d20u, 0x06f3u, 0x1803u, 0x0425u, 0x10c7u, 0x1104u,
    0x1e0eu, 0x0cd3u, 0x0037u, 0x1b9fu, 0x1683u, 0x1685u,
    0x1b09u, 0x1d84u, 0x0a5eu, 0x11e2u, 0x15d9u, 0x1e28u,
    0x0b29u, 0x1402u, 0x1fcfu, 0x132cu, 0x15deu, 0x0000u
));

fn is_gte_q(a: U384) -> bool {
    for (var i = 29u; i < 30u; i = i - 1u) {
        if a.limbs[i] > Q_MODULUS[i] { return true; }
        if a.limbs[i] < Q_MODULUS[i] { return false; }
        if i == 0u { break; }
    }
    return true;
}

fn to_montgomery_u384(a: U384) -> U384 {
    return mul_montgomery_u384(a, R2_MOD_Q);
}
fn from_montgomery_u384(a: U384) -> U384 {
    let one = U384(array<u32, 30>(
        1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
    ));
    return mul_montgomery_u384(a, one);
}
fn normalize_u384(a: U384) -> U384 {
    if is_gte_q(a) {
        return sub_u384(a, U384(Q_MODULUS));
    }
    return a;
}

fn to_montgomery_fp2(a: Fq2) -> Fq2 { return Fq2(to_montgomery_u384(a.c0), to_montgomery_u384(a.c1)); }
fn from_montgomery_fp2(a: Fq2) -> Fq2 { return Fq2(from_montgomery_u384(a.c0), from_montgomery_u384(a.c1)); }
fn normalize_fp2(a: Fq2) -> Fq2 { return Fq2(normalize_u384(a.c0), normalize_u384(a.c1)); }

fn invert_u384(a: U384) -> U384 {
    let q_minus_2 = array<u32, 30>(
        0x0aa9u, 0x1ffdu, 0x1fffu, 0x1dffu, 0x1b9fu, 0x1fffu,
        0x054fu, 0x1fd6u, 0x0bffu, 0x00f5u, 0x1d89u, 0x0d61u,
        0x0a0fu, 0x1869u, 0x1d9cu, 0x0257u, 0x1385u, 0x1c27u,
        0x1dd2u, 0x0ec8u, 0x1acdu, 0x01a5u, 0x1ed9u, 0x0374u,
        0x1a4bu, 0x1f34u, 0x0e5fu, 0x03d4u, 0x0011u, 0x000du
    );
    var res = MONT_ONE;
    var base = a;

    for (var i = 0u; i < 30u; i = i + 1u) {
        var limb = q_minus_2[i];
        for (var j = 0u; j < 13u; j = j + 1u) {
            if (limb & 1u) != 0u { res = mul_montgomery_u384(res, base); }
            base = sqr_montgomery_u384(base);
            limb = limb >> 1u;
        }
    }
    return res;
}

fn invert_fp2(a: Fq2) -> Fq2 {
    let a_sq = sqr_montgomery_u384(a.c0);
    let b_sq = sqr_montgomery_u384(a.c1);
    var denom = add_u384(a_sq, b_sq);
    if is_gte_q(denom) { denom = sub_u384(denom, U384(Q_MODULUS)); }

    let inv_denom = invert_u384(denom);
    let c0 = mul_montgomery_u384(a.c0, inv_denom);
    let neg_c1 = sub_u384(U384(Q_MODULUS), a.c1);
    let c1 = mul_montgomery_u384(neg_c1, inv_denom);
    return Fq2(c0, c1);
}

fn is_inf_g2(p: PointG2) -> bool {
    for (var i = 0u; i < 30u; i = i + 1u) {
        if p.z.c0.limbs[i] != 0u || p.z.c1.limbs[i] != 0u { return false; }
    }
    return true;
}

fn load_g2(p: PointG2) -> PointG2 {
    if is_inf_g2(p) { return G2_INFINITY; }
    return PointG2(to_montgomery_fp2(p.x), to_montgomery_fp2(p.y), to_montgomery_fp2(p.z));
}

fn store_g2(p: PointG2) -> PointG2 {
    if is_inf_g2(p) { return G2_INFINITY; }
    let z_inv = invert_fp2(p.z);
    let z_inv2 = sqr_fp2(z_inv);
    let z_inv3 = mul_fp2(z_inv2, z_inv);

    let x_aff = mul_fp2(p.x, z_inv2);
    let y_aff = mul_fp2(p.y, z_inv3);

    let z_std = Fq2(
        U384(array<u32, 30>(
            1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
        )),
        U384_ZERO
    );
    return PointG2(
        normalize_fp2(from_montgomery_fp2(x_aff)),
        normalize_fp2(from_montgomery_fp2(y_aff)),
        z_std
    );
}

fn store_g2_proj(p: PointG2) -> PointG2 {
    if is_inf_g2(p) { return G2_INFINITY; }
    let z_inv = invert_fp2(p.z);

    let x_aff = mul_fp2(p.x, z_inv);
    let y_aff = mul_fp2(p.y, z_inv);

    let z_std = Fq2(
        U384(array<u32, 30>(
            1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
        )),
        U384_ZERO
    );
    return PointG2(
        normalize_fp2(from_montgomery_fp2(x_aff)),
        normalize_fp2(from_montgomery_fp2(y_aff)),
        z_std
    );
}

struct SubsumParams {
    chunks_per_window: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
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

    var S = G2_PROJ_IDENTITY;
    var running_sum = G2_PROJ_IDENTITY;

    var bucket_ptr = start + count - 1u;
    var next_active_b = bucket_values_g2[bucket_ptr];

    for (var b = next_active_b; b > 0u; b = b - 1u) {
        if b == next_active_b {
            running_sum = add_g2_complete(running_sum, aggregated_buckets_in_g2[bucket_ptr]);
            if bucket_ptr > start {
                bucket_ptr = bucket_ptr - 1u;
                next_active_b = bucket_values_g2[bucket_ptr];
            } else {
                next_active_b = 0u;
            }
        }
        S = add_g2_complete(S, running_sum);
    }

    window_sums_g2[window_id] = store_g2_proj(S);
}

@group(0) @binding(0) var<storage, read> agg_ph1_g2: array<PointG2>;
@group(0) @binding(1) var<storage, read> win_starts_ph1_g2: array<u32>;
@group(0) @binding(2) var<storage, read> win_counts_ph1_g2: array<u32>;
@group(0) @binding(3) var<storage, read_write> partial_sums_g2: array<PointG2>;
@group(0) @binding(4) var<uniform> subsum_params_ph1_g2: SubsumParams;

@compute @workgroup_size(1)
fn subsum_phase1_g2(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let chunks = subsum_params_ph1_g2.chunks_per_window;
    let flat_id = global_id.x;
    let window_id = flat_id / chunks;
    let chunk_id = flat_id % chunks;

    if window_id >= arrayLength(&win_starts_ph1_g2) { return; }

    let start = win_starts_ph1_g2[window_id];
    let count = win_counts_ph1_g2[window_id];

    let chunk_size = (count + chunks - 1u) / chunks;
    let chunk_begin = chunk_id * chunk_size;
    let chunk_end = min(chunk_begin + chunk_size, count);

    var local_sum = G2_PROJ_IDENTITY;
    for (var idx = chunk_begin; idx < chunk_end; idx = idx + 1u) {
        local_sum = add_g2_complete(local_sum, agg_ph1_g2[start + idx]);
    }
    partial_sums_g2[window_id * chunks + chunk_id] = local_sum;
}

@group(0) @binding(0) var<storage, read> partial_sums_ph2_g2: array<PointG2>;
@group(0) @binding(1) var<storage, read_write> win_sums_ph2_g2: array<PointG2>;
@group(0) @binding(2) var<uniform> subsum_params_ph2_g2: SubsumParams;

@compute @workgroup_size(1)
fn subsum_phase2_g2(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let window_id = global_id.x;
    let chunks = subsum_params_ph2_g2.chunks_per_window;

    var sum = G2_PROJ_IDENTITY;
    for (var i = 0u; i < chunks; i = i + 1u) {
        sum = add_g2_complete(sum, partial_sums_ph2_g2[window_id * chunks + i]);
    }
    win_sums_ph2_g2[window_id] = store_g2_proj(sum);
}
