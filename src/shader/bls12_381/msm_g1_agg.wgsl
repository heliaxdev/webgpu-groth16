// src/shader/bls12_381/msm_g1_agg.wgsl

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

fn to_montgomery_u384(a: U384) -> U384 {
    return mul_montgomery_u384(a, R2_MOD_Q);
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

fn add_g1_mixed_safe(p1: PointG1, p2_affine: PointG1) -> PointG1 {
    if is_inf_g1(p1) { return p2_affine; }
    return add_g1_mixed(p1, p2_affine);
}

@group(0) @binding(0) var<storage, read_write> bases_preconv: array<PointG1>;

@compute @workgroup_size(64)
fn to_montgomery_bases_g1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&bases_preconv) { return; }
    let p = bases_preconv[i];
    if is_inf_g1(p) { return; }
    bases_preconv[i] = PointG1(
        to_montgomery_u384(p.x),
        to_montgomery_u384(p.y),
        to_montgomery_u384(p.z)
    );
}

fn load_g1_mont(p: PointG1) -> PointG1 {
    if is_inf_g1(p) { return G1_INFINITY; }
    return p;
}

@group(0) @binding(0) var<storage, read> bases_g1: array<PointG1>;
@group(0) @binding(1) var<storage, read> base_indices: array<u32>;
@group(0) @binding(2) var<storage, read> bucket_pointers: array<u32>;
@group(0) @binding(3) var<storage, read> bucket_sizes: array<u32>;
@group(0) @binding(4) var<storage, read_write> aggregated_buckets_g1: array<PointG1>;
@group(0) @binding(5) var<storage, read> bucket_values_agg: array<u32>;

fn negate_g1(p: PointG1) -> PointG1 {
    return PointG1(p.x, sub_u384(U384(Q_MODULUS), p.y), p.z);
}

@compute @workgroup_size(64)
fn aggregate_buckets_g1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bucket_idx = global_id.x;
    if bucket_idx >= arrayLength(&bucket_pointers) { return; }

    let start = bucket_pointers[bucket_idx];
    let size = bucket_sizes[bucket_idx];
    var sum = G1_INFINITY;
    for (var i = 0u; i < size; i = i + 1u) {
        let raw = base_indices[start + i];
        let point_idx = raw & 0x7FFFFFFFu;
        let is_neg = (raw >> 31u) != 0u;
        var base = load_g1_mont(bases_g1[point_idx]);
        if is_neg {
            base = negate_g1(base);
        }
        sum = add_g1_mixed_safe(sum, base);
    }
    aggregated_buckets_g1[bucket_idx] = sum;
}

@group(0) @binding(0) var<storage, read> reduce_input_g1: array<PointG1>;
@group(0) @binding(1) var<storage, read> reduce_starts_g1: array<u32>;
@group(0) @binding(2) var<storage, read> reduce_counts_g1: array<u32>;
@group(0) @binding(3) var<storage, read_write> reduce_output_g1: array<PointG1>;

@compute @workgroup_size(64)
fn reduce_sub_buckets_g1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let parent_idx = global_id.x;
    if parent_idx >= arrayLength(&reduce_starts_g1) { return; }

    let start = reduce_starts_g1[parent_idx];
    let count = reduce_counts_g1[parent_idx];

    if count == 1u {
        reduce_output_g1[parent_idx] = reduce_input_g1[start];
        return;
    }

    var sum = reduce_input_g1[start];
    for (var i = 1u; i < count; i = i + 1u) {
        sum = add_g1_safe(sum, reduce_input_g1[start + i]);
    }
    reduce_output_g1[parent_idx] = sum;
}

fn scalar_mul_g1(p: PointG1, k: u32) -> PointG1 {
    if k == 0u { return G1_INFINITY; }
    if k == 1u { return p; }
    var result = G1_INFINITY;
    var base = p;
    var scalar = k;
    for (var bit = 0u; bit < 16u; bit = bit + 1u) {
        if scalar == 0u { break; }
        if (scalar & 1u) != 0u {
            result = add_g1_safe(result, base);
        }
        base = double_g1(base);
        scalar = scalar >> 1u;
    }
    return result;
}

@group(0) @binding(0) var<storage, read_write> weight_buckets_g1_data: array<PointG1>;
@group(0) @binding(1) var<storage, read> weight_bucket_values: array<u32>;

@compute @workgroup_size(64)
fn weight_buckets_g1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&weight_buckets_g1_data) { return; }
    weight_buckets_g1_data[i] = scalar_mul_g1(weight_buckets_g1_data[i], weight_bucket_values[i]);
}
