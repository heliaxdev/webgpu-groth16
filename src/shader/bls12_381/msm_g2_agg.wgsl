// src/shader/bls12_381/msm_g2_agg.wgsl

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

fn to_montgomery_u384(a: U384) -> U384 {
    return mul_montgomery_u384(a, R2_MOD_Q);
}

fn to_montgomery_fp2(a: Fq2) -> Fq2 { return Fq2(to_montgomery_u384(a.c0), to_montgomery_u384(a.c1)); }

fn is_inf_g2(p: PointG2) -> bool {
    for (var i = 0u; i < 30u; i = i + 1u) {
        if p.z.c0.limbs[i] != 0u || p.z.c1.limbs[i] != 0u { return false; }
    }
    return true;
}

fn add_g2_safe(p1: PointG2, p2: PointG2) -> PointG2 {
    let p1_inf = is_inf_g2(p1); let p2_inf = is_inf_g2(p2);
    if p1_inf && p2_inf { return G2_INFINITY; }
    if p1_inf { return p2; } if p2_inf { return p1; }
    return add_g2(p1, p2);
}

fn add_g2_mixed_safe(p1: PointG2, p2_affine: PointG2) -> PointG2 {
    if is_inf_g2(p1) { return p2_affine; }
    return add_g2_mixed(p1, p2_affine);
}

@group(0) @binding(0) var<storage, read_write> bases_preconv_g2: array<PointG2>;

@compute @workgroup_size(64)
fn to_montgomery_bases_g2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&bases_preconv_g2) { return; }
    let p = bases_preconv_g2[i];
    if is_inf_g2(p) { return; }
    bases_preconv_g2[i] = PointG2(
        to_montgomery_fp2(p.x),
        to_montgomery_fp2(p.y),
        to_montgomery_fp2(p.z)
    );
}

fn load_g2_mont(p: PointG2) -> PointG2 {
    if is_inf_g2(p) { return G2_INFINITY; }
    return p;
}

@group(0) @binding(0) var<storage, read> bases_g2: array<PointG2>;
@group(0) @binding(1) var<storage, read> base_indices_g2: array<u32>;
@group(0) @binding(2) var<storage, read> bucket_pointers_g2: array<u32>;
@group(0) @binding(3) var<storage, read> bucket_sizes_g2: array<u32>;
@group(0) @binding(4) var<storage, read_write> aggregated_buckets_g2: array<PointG2>;
@group(0) @binding(5) var<storage, read> bucket_values_agg_g2: array<u32>;

fn negate_g2(p: PointG2) -> PointG2 {
    return PointG2(
        p.x,
        Fq2(sub_u384(U384(Q_MODULUS), p.y.c0), sub_u384(U384(Q_MODULUS), p.y.c1)),
        p.z
    );
}

@compute @workgroup_size(64)
fn aggregate_buckets_g2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bucket_idx = global_id.x;
    if bucket_idx >= arrayLength(&bucket_pointers_g2) { return; }

    let start = bucket_pointers_g2[bucket_idx];
    let size = bucket_sizes_g2[bucket_idx];
    var sum = G2_INFINITY;
    for (var i = 0u; i < size; i = i + 1u) {
        let raw = base_indices_g2[start + i];
        let point_idx = raw & 0x7FFFFFFFu;
        let is_neg = (raw >> 31u) != 0u;
        var base = load_g2_mont(bases_g2[point_idx]);
        if is_neg {
            base = negate_g2(base);
        }
        sum = add_g2_mixed_safe(sum, base);
    }
    aggregated_buckets_g2[bucket_idx] = sum;
}

@group(0) @binding(0) var<storage, read> reduce_input_g2: array<PointG2>;
@group(0) @binding(1) var<storage, read> reduce_starts_g2: array<u32>;
@group(0) @binding(2) var<storage, read> reduce_counts_g2: array<u32>;
@group(0) @binding(3) var<storage, read_write> reduce_output_g2: array<PointG2>;

@compute @workgroup_size(64)
fn reduce_sub_buckets_g2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let parent_idx = global_id.x;
    if parent_idx >= arrayLength(&reduce_starts_g2) { return; }

    let start = reduce_starts_g2[parent_idx];
    let count = reduce_counts_g2[parent_idx];

    if count == 1u {
        reduce_output_g2[parent_idx] = reduce_input_g2[start];
        return;
    }

    var sum = reduce_input_g2[start];
    for (var i = 1u; i < count; i = i + 1u) {
        sum = add_g2_safe(sum, reduce_input_g2[start + i]);
    }
    reduce_output_g2[parent_idx] = sum;
}

fn jacobian_to_proj_g2(p: PointG2) -> PointG2 {
    if is_inf_g2(p) { return G2_PROJ_IDENTITY; }
    let x_proj = mul_fp2(p.x, p.z);
    let z_sq = sqr_fp2(p.z);
    let z_proj = mul_fp2(z_sq, p.z);
    return PointG2(x_proj, p.y, z_proj);
}

fn scalar_mul_g2(p: PointG2, k: u32) -> PointG2 {
    if k == 0u { return G2_PROJ_IDENTITY; }
    if k == 1u { return p; }
    var result = G2_PROJ_IDENTITY;
    var base = p;
    var scalar = k;
    for (var bit = 0u; bit < 16u; bit = bit + 1u) {
        if scalar == 0u { break; }
        if (scalar & 1u) != 0u {
            result = add_g2_complete(result, base);
        }
        base = add_g2_complete(base, base);
        scalar = scalar >> 1u;
    }
    return result;
}

@group(0) @binding(0) var<storage, read_write> weight_buckets_g2_data: array<PointG2>;
@group(0) @binding(1) var<storage, read> weight_bucket_values_g2: array<u32>;

@compute @workgroup_size(64)
fn weight_buckets_g2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&weight_buckets_g2_data) { return; }
    let p = jacobian_to_proj_g2(weight_buckets_g2_data[i]);
    weight_buckets_g2_data[i] = scalar_mul_g2(p, weight_bucket_values_g2[i]);
}
