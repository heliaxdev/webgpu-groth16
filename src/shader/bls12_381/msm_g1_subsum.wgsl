// src/shader/bls12_381/msm_g1_subsum.wgsl

const U384_ZERO = U384(array<u32, 30>(
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
));

const G1_INFINITY = PointG1(U384_ZERO, U384_ZERO, U384_ZERO);
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

fn to_montgomery_u384(a: U384) -> U384 {
    return mul_montgomery_u384(a, R2_MOD_Q);
}

fn is_gte_q(a: U384) -> bool {
    for (var i = 29u; i < 30u; i = i - 1u) {
        if a.limbs[i] > Q_MODULUS[i] { return true; }
        if a.limbs[i] < Q_MODULUS[i] { return false; }
        if i == 0u { break; }
    }
    return true;
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

fn is_inf_g1(p: PointG1) -> bool {
    for (var i = 0u; i < 30u; i = i + 1u) {
        if p.z.limbs[i] != 0u { return false; }
    }
    return true;
}

fn add_g1_safe(p1: PointG1, p2: PointG1) -> PointG1 {
    let p1_inf = is_inf_g1(p1); let p2_inf = is_inf_g1(p2);
    if p1_inf && p2_inf { return G1_INFINITY; }
    if p1_inf { return p2; } if p2_inf { return p1; }
    return add_g1(p1, p2);
}

fn load_g1(p: PointG1) -> PointG1 {
    if is_inf_g1(p) { return G1_INFINITY; }
    return PointG1(mul_montgomery_u384(p.x, R2_MOD_Q), mul_montgomery_u384(p.y, R2_MOD_Q), mul_montgomery_u384(p.z, R2_MOD_Q));
}

fn store_g1(p: PointG1) -> PointG1 {
    if is_inf_g1(p) { return G1_INFINITY; }
    let z_inv = invert_u384(p.z);
    let z_inv2 = sqr_montgomery_u384(z_inv);
    let z_inv3 = mul_montgomery_u384(z_inv2, z_inv);

    let x_aff = mul_montgomery_u384(p.x, z_inv2);
    let y_aff = mul_montgomery_u384(p.y, z_inv3);

    let z_std = U384(array<u32, 30>(
        1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
    ));
    return PointG1(
        normalize_u384(from_montgomery_u384(x_aff)),
        normalize_u384(from_montgomery_u384(y_aff)),
        z_std
    );
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

    var sum = G1_INFINITY;
    for (var i = 0u; i < count; i = i + 1u) {
        sum = add_g1_safe(sum, aggregated_buckets_in_g1[start + i]);
    }
    window_sums_g1[window_id] = store_g1(sum);
}

struct SubsumParams {
    chunks_per_window: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

const G1_SUBSUM_WG_SIZE: u32 = 64u;
var<workgroup> subsum_shared_g1: array<PointG1, 64>;

@group(0) @binding(0) var<storage, read> agg_ph1_g1: array<PointG1>;
@group(0) @binding(1) var<storage, read> win_starts_ph1: array<u32>;
@group(0) @binding(2) var<storage, read> win_counts_ph1: array<u32>;
@group(0) @binding(3) var<storage, read_write> partial_sums_g1: array<PointG1>;
@group(0) @binding(4) var<uniform> subsum_params_ph1: SubsumParams;

@compute @workgroup_size(64)
fn subsum_phase1_g1(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let window_id = wg_id.x;
    let tid = local_id.x;
    if window_id >= arrayLength(&win_starts_ph1) { return; }

    let start = win_starts_ph1[window_id];
    let count = win_counts_ph1[window_id];

    var local_sum = G1_INFINITY;
    var i = tid;
    for (var iter = 0u; iter < 65536u; iter = iter + 1u) {
        if i >= count { break; }
        local_sum = add_g1_safe(local_sum, agg_ph1_g1[start + i]);
        i = i + G1_SUBSUM_WG_SIZE;
    }
    subsum_shared_g1[tid] = local_sum;
    workgroupBarrier();

    if tid < 32u { subsum_shared_g1[tid] = add_g1_safe(subsum_shared_g1[tid], subsum_shared_g1[tid + 32u]); }
    workgroupBarrier();
    if tid < 16u { subsum_shared_g1[tid] = add_g1_safe(subsum_shared_g1[tid], subsum_shared_g1[tid + 16u]); }
    workgroupBarrier();
    if tid < 8u { subsum_shared_g1[tid] = add_g1_safe(subsum_shared_g1[tid], subsum_shared_g1[tid + 8u]); }
    workgroupBarrier();
    if tid < 4u { subsum_shared_g1[tid] = add_g1_safe(subsum_shared_g1[tid], subsum_shared_g1[tid + 4u]); }
    workgroupBarrier();
    if tid < 2u { subsum_shared_g1[tid] = add_g1_safe(subsum_shared_g1[tid], subsum_shared_g1[tid + 2u]); }
    workgroupBarrier();
    if tid == 0u {
        let result = add_g1_safe(subsum_shared_g1[0], subsum_shared_g1[1]);
        partial_sums_g1[window_id] = store_g1(result);
    }
}

@group(0) @binding(0) var<storage, read> partial_sums_ph2_g1: array<PointG1>;
@group(0) @binding(1) var<storage, read_write> win_sums_ph2_g1: array<PointG1>;
@group(0) @binding(2) var<uniform> subsum_params_ph2: SubsumParams;

@compute @workgroup_size(1)
fn subsum_phase2_g1(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let window_id = global_id.x;
    win_sums_ph2_g1[window_id] = partial_sums_ph2_g1[window_id];
}
