// src/shader/bls12_381/msm.wgsl

// ============================================================================
// CONSTANTS & MONTGOMERY CONVERSION (30 × 13-bit limbs, R = 2^390)
// ============================================================================

const U384_ZERO = U384(array<u32, 30>(
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
));

const G1_INFINITY = PointG1(U384_ZERO, U384_ZERO, U384_ZERO);

const FQ2_ZERO = Fq2(U384_ZERO, U384_ZERO);
const G2_INFINITY = PointG2(FQ2_ZERO, FQ2_ZERO, FQ2_ZERO);

// Projective identity for G2: (0:1:0) in Montgomery form.
// Used with add_g2_complete which operates in projective coordinates.
const FQ2_ONE_MONT = Fq2(MONT_ONE, U384_ZERO);
const G2_PROJ_IDENTITY = PointG2(FQ2_ZERO, FQ2_ONE_MONT, FQ2_ZERO);

fn is_gte_q(a: U384) -> bool {
    for (var i = 29u; i < 30u; i = i - 1u) {
        if a.limbs[i] > Q_MODULUS[i] { return true; }
        if a.limbs[i] < Q_MODULUS[i] { return false; }
        if i == 0u { break; }
    }
    return true;
}

// R^2 mod q for BLS12-381 base field F_q (30 × 13-bit limbs, R = 2^390).
const R2_MOD_Q = U384(array<u32, 30>(
    0x070fu, 0x0880u, 0x10d1u, 0x0c83u, 0x1aecu, 0x1121u,
    0x004cu, 0x1874u, 0x066eu, 0x1b75u, 0x01ebu, 0x1beau,
    0x07b1u, 0x1f70u, 0x117bu, 0x0362u, 0x0ed2u, 0x090fu,
    0x110au, 0x1482u, 0x0f70u, 0x1699u, 0x05dcu, 0x1200u,
    0x0c97u, 0x0c8cu, 0x12b3u, 0x1dc0u, 0x1696u, 0x0007u
));

// Montgomery representation of 1: R mod q (30 × 13-bit limbs)
const MONT_ONE = U384(array<u32, 30>(
    0x1f2eu, 0x068fu, 0x0000u, 0x0c00u, 0x0467u, 0x0056u,
    0x0d20u, 0x06f3u, 0x1803u, 0x0425u, 0x10c7u, 0x1104u,
    0x1e0eu, 0x0cd3u, 0x0037u, 0x1b9fu, 0x1683u, 0x1685u,
    0x1b09u, 0x1d84u, 0x0a5eu, 0x11e2u, 0x15d9u, 0x1e28u,
    0x0b29u, 0x1402u, 0x1fcfu, 0x132cu, 0x15deu, 0x0000u
));

// Computes a * R mod q, putting standard input into Montgomery form.
fn to_montgomery_u384(a: U384) -> U384 {
    return mul_montgomery_u384(a, R2_MOD_Q);
}

// Converts Montgomery form back to standard form
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

// Inverts a Montgomery U384 via Fermat's Little Theorem (a^(q-2) mod q)
fn invert_u384(a: U384) -> U384 {
    // q - 2 in 30 × 13-bit limbs
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

fn to_montgomery_fp2(a: Fq2) -> Fq2 { return Fq2(to_montgomery_u384(a.c0), to_montgomery_u384(a.c1)); }
fn from_montgomery_fp2(a: Fq2) -> Fq2 { return Fq2(from_montgomery_u384(a.c0), from_montgomery_u384(a.c1)); }
fn normalize_fp2(a: Fq2) -> Fq2 { return Fq2(normalize_u384(a.c0), normalize_u384(a.c1)); }

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

// ============================================================================
// DATA LOADERS & DOMAIN BRIDGES
// ============================================================================

fn is_inf_g1(p: PointG1) -> bool {
    for (var i = 0u; i < 30u; i = i + 1u) {
        if p.z.limbs[i] != 0u { return false; }
    }
    return true;
}

fn is_inf_g2(p: PointG2) -> bool {
    for (var i = 0u; i < 30u; i = i + 1u) {
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

// Mixed safe: P1 projective + P2 affine (Z2 = R in Montgomery form).
// P2 is never infinity (bucket sorting filters zeros), so only check P1.
fn add_g1_mixed_safe(p1: PointG1, p2_affine: PointG1) -> PointG1 {
    if is_inf_g1(p1) { return p2_affine; }
    return add_g1_mixed(p1, p2_affine);
}

fn add_g2_safe(p1: PointG2, p2: PointG2) -> PointG2 {
    let p1_inf = is_inf_g2(p1); let p2_inf = is_inf_g2(p2);
    if p1_inf && p2_inf { return G2_INFINITY; }
    if p1_inf { return p2; } if p2_inf { return p1; }
    return add_g2(p1, p2);
}

// Mixed safe: P1 projective + P2 affine (Z2 = (R,0) in Montgomery form).
// P2 is never infinity (bucket sorting filters zeros), so only check P1.
fn add_g2_mixed_safe(p1: PointG2, p2_affine: PointG2) -> PointG2 {
    if is_inf_g2(p1) { return p2_affine; }
    return add_g2_mixed(p1, p2_affine);
}

// Load Standard Affine -> Montgomery Jacobian
fn load_g1(p: PointG1) -> PointG1 {
    if is_inf_g1(p) { return G1_INFINITY; }
    return PointG1(to_montgomery_u384(p.x), to_montgomery_u384(p.y), to_montgomery_u384(p.z));
}

// Convert Montgomery Jacobian -> Standard Affine (to return to CPU)
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

// Convert PROJECTIVE Montgomery → Standard Affine: x = X/Z, y = Y/Z.
// Used by add_g2_complete pipeline (projective coords, not Jacobian).
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

// ============================================================================
// BASE POINT PRE-CONVERSION (standard → Montgomery form)
// ============================================================================

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

// Load a base point that is already in Montgomery form (skip conversion).
fn load_g1_mont(p: PointG1) -> PointG1 {
    if is_inf_g1(p) { return G1_INFINITY; }
    return p;
}

// G2 base pre-conversion: standard affine -> Montgomery form (in-place).
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

// Load a G2 base point that is already in Montgomery form (skip conversion).
fn load_g2_mont(p: PointG2) -> PointG2 {
    if is_inf_g2(p) { return G2_INFINITY; }
    return p;
}

// ============================================================================
// PIPPENGER ALGORITHM PIPELINES (G1 and G2)
// ============================================================================

@group(0) @binding(0) var<storage, read> bases_g1: array<PointG1>;
@group(0) @binding(1) var<storage, read> base_indices: array<u32>;
@group(0) @binding(2) var<storage, read> bucket_pointers: array<u32>;
@group(0) @binding(3) var<storage, read> bucket_sizes: array<u32>;
@group(0) @binding(4) var<storage, read_write> aggregated_buckets_g1: array<PointG1>;
@group(0) @binding(5) var<storage, read> bucket_values_agg: array<u32>;

// Negate a G1 point in Montgomery form: (x, y, z) → (x, q−y, z).
fn negate_g1(p: PointG1) -> PointG1 {
    return PointG1(p.x, sub_u384(U384(Q_MODULUS), p.y), p.z);
}

// Phase 1a: Sum points in each bucket (no weighting).
// Kept separate from weighting to avoid Metal shader miscompilation
// when add_g1_mixed and scalar_mul_g1 are compiled in the same kernel.
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

// Phase 1a-reduce: Sum sub-buckets of the same parent into a single bucket result.
// Used when large buckets are split into capped-size sub-buckets for load balancing.
// Thread i handles original bucket i: reads reduce_starts[i]..+reduce_counts[i] from
// the sub-bucket intermediate buffer and writes the sum to the output buffer.
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

// Computes k * P using double-and-add.
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

// Phase 1b: Weight each bucket sum by its bucket value: v * B[v].
// Separate kernel to avoid Metal shader miscompilation (see aggregate_buckets_g1).
@group(0) @binding(0) var<storage, read_write> weight_buckets_g1_data: array<PointG1>;
@group(0) @binding(1) var<storage, read> weight_bucket_values: array<u32>;

@compute @workgroup_size(64)
fn weight_buckets_g1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&weight_buckets_g1_data) { return; }
    weight_buckets_g1_data[i] = scalar_mul_g1(weight_buckets_g1_data[i], weight_bucket_values[i]);
}

// ==== Suffix Scan + Gap-Weight Pipeline (G1) ====
//
@group(0) @binding(0) var<storage, read> aggregated_buckets_in_g1: array<PointG1>;
@group(0) @binding(1) var<storage, read> bucket_values: array<u32>;
@group(0) @binding(2) var<storage, read> window_starts: array<u32>;
@group(0) @binding(3) var<storage, read> window_counts: array<u32>;
@group(0) @binding(4) var<storage, read_write> window_sums_g1: array<PointG1>;

// Single-thread G1 subsum accumulation.
// With 30×13-bit limbs, PointG1 = 360 bytes which triggers Metal
// var<workgroup> corruption. Use workgroup_size(1) without shared memory.
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

// ==== G1 Parallel Shared-Memory Tree Reduction ====
//
// Single-pass: one workgroup of 64 threads per window. Each thread sums a
// strided subset of weighted buckets, then a 6-stage binary tree reduction
// in var<workgroup> memory produces the final window sum.
//
// The @size(128) padding on PointG1 members ensures 16-byte-aligned array
// strides in Metal threadgroup memory (384 % 16 = 0), fixing the data
// corruption that occurred with the unpadded 360-byte layout.

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

    // Phase 1: Each thread sums a strided subset of weighted buckets.
    var local_sum = G1_INFINITY;
    var i = tid;
    for (var iter = 0u; iter < 65536u; iter = iter + 1u) {
        if i >= count { break; }
        local_sum = add_g1_safe(local_sum, agg_ph1_g1[start + i]);
        i = i + G1_SUBSUM_WG_SIZE;
    }
    subsum_shared_g1[tid] = local_sum;
    workgroupBarrier();

    // Phase 2: 6-stage binary tree reduction in shared memory.
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

// G1 Phase 2 is now a no-op identity copy since Phase 1 already produces
// final window sums. Kept for pipeline compatibility.
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

// ==== G2 Pipelines ====

@group(0) @binding(0) var<storage, read> bases_g2: array<PointG2>;
@group(0) @binding(1) var<storage, read> base_indices_g2: array<u32>;
@group(0) @binding(2) var<storage, read> bucket_pointers_g2: array<u32>;
@group(0) @binding(3) var<storage, read> bucket_sizes_g2: array<u32>;
@group(0) @binding(4) var<storage, read_write> aggregated_buckets_g2: array<PointG2>;
@group(0) @binding(5) var<storage, read> bucket_values_agg_g2: array<u32>;

// Negate a G2 point in Montgomery form: (x, y, z) → (x, −y, z) over Fq2.
fn negate_g2(p: PointG2) -> PointG2 {
    return PointG2(
        p.x,
        Fq2(sub_u384(U384(Q_MODULUS), p.y.c0), sub_u384(U384(Q_MODULUS), p.y.c1)),
        p.z
    );
}

// Aggregate uses fast Jacobian mixed addition (base points are affine with Z=(R,0)).
// Output is in JACOBIAN coordinates; weight_buckets_g2 converts to projective inline.
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

// Phase 1a-reduce for G2: Sum sub-buckets of the same parent into a single bucket result.
// Aggregate outputs Jacobian; add_g2 calls double_g2 which has Metal issues under pressure.
// Use add_g2 here — the reduce kernel has low register pressure (simple loop only).
// NOTE: add_g2_safe is defined earlier in this file (used by aggregate_buckets_g2 too).
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

@group(0) @binding(0) var<storage, read> aggregated_buckets_in_g2: array<PointG2>;
@group(0) @binding(1) var<storage, read> bucket_values_g2: array<u32>;
@group(0) @binding(2) var<storage, read> window_starts_g2: array<u32>;
@group(0) @binding(3) var<storage, read> window_counts_g2: array<u32>;
@group(0) @binding(4) var<storage, read_write> window_sums_g2: array<PointG2>;

// Legacy running-sum subsum (NOT dispatched — tree reduction replaced it).
// NOTE: This reads Jacobian output from aggregate but uses add_g2_complete (projective).
// Would need jacobian_to_proj_g2 conversion if ever re-enabled.
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

// ==== G2 Projective Pipeline: Weight + Tree Reduction ====
//
// Uses add_g2_complete (projective, no double_g2) throughout.
// Aggregate outputs Jacobian; weight_buckets_g2 converts inline before scalar_mul.

// Convert Jacobian (X,Y,Z) → Projective (X*Z, Y, Z³).
// Jacobian: affine = (X/Z², Y/Z³). Projective: affine = (X'/Z', Y'/Z').
// So X' = X*Z, Y' = Y, Z' = Z³ gives X'/Z' = X/Z², Y'/Z' = Y/Z³.
fn jacobian_to_proj_g2(p: PointG2) -> PointG2 {
    if is_inf_g2(p) { return G2_PROJ_IDENTITY; }
    let x_proj = mul_fp2(p.x, p.z);       // X * Z
    let z_sq = sqr_fp2(p.z);
    let z_proj = mul_fp2(z_sq, p.z);       // Z³
    return PointG2(x_proj, p.y, z_proj);
}

// Computes k * P using double-and-add via add_g2_complete.
// No separate doubling function — doubling is add_g2_complete(P, P).
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

// Phase 1b: Weight each bucket sum by its bucket value: v * B[v].
@group(0) @binding(0) var<storage, read_write> weight_buckets_g2_data: array<PointG2>;
@group(0) @binding(1) var<storage, read> weight_bucket_values_g2: array<u32>;

@compute @workgroup_size(64)
fn weight_buckets_g2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&weight_buckets_g2_data) { return; }
    // Aggregate outputs Jacobian; convert to projective before scalar mul.
    let p = jacobian_to_proj_g2(weight_buckets_g2_data[i]);
    weight_buckets_g2_data[i] = scalar_mul_g2(p, weight_bucket_values_g2[i]);
}

// Phase 1 of tree reduction: split each window's buckets into chunks, sum each chunk.
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

// Phase 2: reduce partial sums into final window sums, converting to standard affine.
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

// ==== Debug round-trip kernels (test-only usage from Rust) ====

@group(0) @binding(0) var<storage, read> rt_in_g1: array<PointG1>;
@group(0) @binding(1) var<storage, read_write> rt_out_g1: array<PointG1>;

@compute @workgroup_size(64)
fn roundtrip_g1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&rt_in_g1) { return; }
    rt_out_g1[i] = store_g1(load_g1(rt_in_g1[i]));
}

// Debug: load a point, double it, and store it back to standard affine.
@compute @workgroup_size(1)
fn roundtrip_double_g1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&rt_in_g1) { return; }
    let p = load_g1(rt_in_g1[i]);
    let doubled = double_g1(p);
    rt_out_g1[i] = store_g1(doubled);
}

@group(0) @binding(0) var<storage, read> rt_in_g2: array<PointG2>;
@group(0) @binding(1) var<storage, read_write> rt_out_g2: array<PointG2>;

@compute @workgroup_size(64)
fn roundtrip_g2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&rt_in_g2) { return; }
    rt_out_g2[i] = store_g2(load_g2(rt_in_g2[i]));
}

// Debug: load two G2 points, add with add_g2_complete, store result (projective→affine).
@group(0) @binding(0) var<storage, read> rt_add_g2_in_a: array<PointG2>;
@group(0) @binding(1) var<storage, read> rt_add_g2_in_b: array<PointG2>;
@group(0) @binding(2) var<storage, read_write> rt_add_g2_out: array<PointG2>;

@compute @workgroup_size(1)
fn roundtrip_add_g2_complete(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&rt_add_g2_in_a) { return; }
    // Load points into Montgomery projective form.
    // Input Z = (1,0) standard → (R,0) Montgomery = projective Z=1.
    let a = load_g2(rt_add_g2_in_a[i]);
    let b = load_g2(rt_add_g2_in_b[i]);
    let sum = add_g2_complete(a, b);
    rt_add_g2_out[i] = store_g2_proj(sum);
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

// ==== Workgroup Memory Diagnostic Test ====
//
// Tests whether var<workgroup> with PointG1 works correctly on the current GPU.
// Uses a 64-thread parallel tree reduction in shared memory.

@group(0) @binding(0) var<storage, read> wg_test_in_g1: array<PointG1>;
@group(0) @binding(1) var<storage, read_write> wg_test_out_g1: array<PointG1>;

const WG_TEST_SIZE: u32 = 64u;
var<workgroup> wg_test_shared: array<PointG1, 64>;

@compute @workgroup_size(64)
fn test_workgroup_reduction_g1(
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;

    // Load point into Montgomery Jacobian, store into shared memory.
    wg_test_shared[tid] = load_g1(wg_test_in_g1[tid]);
    workgroupBarrier();

    // 6-stage binary tree reduction.
    if tid < 32u { wg_test_shared[tid] = add_g1_safe(wg_test_shared[tid], wg_test_shared[tid + 32u]); }
    workgroupBarrier();
    if tid < 16u { wg_test_shared[tid] = add_g1_safe(wg_test_shared[tid], wg_test_shared[tid + 16u]); }
    workgroupBarrier();
    if tid < 8u { wg_test_shared[tid] = add_g1_safe(wg_test_shared[tid], wg_test_shared[tid + 8u]); }
    workgroupBarrier();
    if tid < 4u { wg_test_shared[tid] = add_g1_safe(wg_test_shared[tid], wg_test_shared[tid + 4u]); }
    workgroupBarrier();
    if tid < 2u { wg_test_shared[tid] = add_g1_safe(wg_test_shared[tid], wg_test_shared[tid + 2u]); }
    workgroupBarrier();
    if tid == 0u {
        let result = add_g1_safe(wg_test_shared[0], wg_test_shared[1]);
        wg_test_out_g1[0] = store_g1(result);
    }
}
