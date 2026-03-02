// src/shader/bls12_381/ntt.wgsl

// ============================================================================
// CONSTANTS & WORKGROUP SIZE
// ============================================================================
const THREADS_PER_WORKGROUP: u32 = 256u;
const ELEMENTS_PER_TILE: u32 = 512u;

// ============================================================================
// GPU PIPELINE BUFFERS
// ============================================================================

@group(0) @binding(0)
var<storage, read_write> data: array<U256>;

@group(0) @binding(1)
var<storage, read> twiddles: array<U256>;

struct NttParams {
    n: u32,
    half_len: u32,
    log_n: u32,
    _pad: u32,
}

@group(0) @binding(2)
var<uniform> params: NttParams;

@group(2) @binding(0)
var<storage, read> pointwise_a: array<U256>;

@group(2) @binding(1)
var<storage, read> pointwise_b: array<U256>;

@group(2) @binding(2)
var<storage, read> pointwise_c: array<U256>;

@group(2) @binding(3)
var<storage, read> pointwise_z_inv: array<U256>;

// Fast workgroup-shared memory arrays
var<workgroup> shared_data: array<U256, ELEMENTS_PER_TILE>;
var<workgroup> shared_twiddles: array<U256, THREADS_PER_WORKGROUP>;

fn add_fr(a: U256, b: U256) -> U256 {
    var sum = add_u256(a, b);
    var is_gte = true;
    for (var i: u32 = 7u; i < 8u; i = i - 1u) {
        if sum.limbs[i] > R_MODULUS[i] { break; }
        if sum.limbs[i] < R_MODULUS[i] { is_gte = false; break; }
        if i == 0u { break; }
    }
    if is_gte {
        sum = sub_u256(sum, U256(R_MODULUS));
    }
    return sum;
}

fn sub_fr(a: U256, b: U256) -> U256 {
    var is_less = false;
    for (var i: u32 = 7u; i < 8u; i = i - 1u) {
        if a.limbs[i] < b.limbs[i] { is_less = true; break; }
        if a.limbs[i] > b.limbs[i] { break; }
        if i == 0u { break; }
    }

    var diff = sub_u256(a, b);
    if is_less {
        diff = add_u256(diff, U256(R_MODULUS));
    }
    return diff;
}

// ============================================================================
// HELPER: BIT REVERSAL
// ============================================================================

// Reverses the `bit_len` lower bits of `n`.
fn reverse_bits(n: u32, bit_len: u32) -> u32 {
    return reverseBits(n) >> (32u - bit_len);
}

fn compute_pointwise_h(i: u32) -> U256 {
    let a = pointwise_a[i];
    let b = pointwise_b[i];
    let c = pointwise_c[i];
    let ab = mul_montgomery_u256(a, b);
    let ab_c = sub_fr(ab, c);
    return mul_montgomery_u256(ab_c, pointwise_z_inv[0u]);
}

@compute @workgroup_size(256)
fn bitreverse_inplace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = params.n;
    if i >= n {
        return;
    }

    let j = reverse_bits(i, params.log_n);
    if i < j {
        let tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

@compute @workgroup_size(256)
fn bitreverse_fused_pointwise(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = params.n;
    if i >= n {
        return;
    }

    let j = reverse_bits(i, params.log_n);
    data[j] = compute_pointwise_h(i);
}

@compute @workgroup_size(256)
fn ntt_global_stage(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let n = params.n;
    let half_len = params.half_len;
    if half_len == 0u {
        return;
    }

    let butterflies = n / 2u;
    if tid >= butterflies {
        return;
    }

    let len = half_len * 2u;
    let k = tid % half_len;
    let pos = (tid / half_len) * len + k;
    let twiddle_stride = n / len;
    let twiddle = twiddles[k * twiddle_stride];

    let u = data[pos];
    let v = data[pos + half_len];
    let v_omega = mul_montgomery_u256(v, twiddle);

    data[pos] = add_fr(u, v_omega);
    data[pos + half_len] = sub_fr(u, v_omega);
}

@compute @workgroup_size(256)
fn ntt_global_stage_dif(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let n = params.n;
    let half_len = params.half_len;
    if half_len == 0u {
        return;
    }

    let butterflies = n / 2u;
    if tid >= butterflies {
        return;
    }

    let len = half_len * 2u;
    let k = tid % half_len;
    let pos = (tid / half_len) * len + k;
    let twiddle_stride = n / len;
    let twiddle = twiddles[k * twiddle_stride];

    let u = data[pos];
    let v = data[pos + half_len];
    let sum = add_fr(u, v);
    let diff = sub_fr(u, v);

    data[pos] = sum;
    data[pos + half_len] = mul_montgomery_u256(diff, twiddle);
}

@compute @workgroup_size(256)
fn ntt_global_stage_dif_fused_pointwise(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let n = params.n;
    let half_len = params.half_len;
    if half_len == 0u {
        return;
    }

    let butterflies = n / 2u;
    if tid >= butterflies {
        return;
    }

    let len = half_len * 2u;
    let k = tid % half_len;
    let pos = (tid / half_len) * len + k;
    let twiddle_stride = n / len;
    let twiddle = twiddles[k * twiddle_stride];

    let u = compute_pointwise_h(pos);
    let v = compute_pointwise_h(pos + half_len);
    let sum = add_fr(u, v);
    let diff = sub_fr(u, v);

    data[pos] = sum;
    data[pos + half_len] = mul_montgomery_u256(diff, twiddle);
}

@compute @workgroup_size(256)
fn ntt_global_stage_radix4(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let n = params.n;
    let half_len = params.half_len;
    if half_len == 0u {
        return;
    }

    let butterflies = n / 4u;
    if tid >= butterflies {
        return;
    }

    let len4 = half_len * 4u;
    let k = tid % half_len;
    let pos = (tid / half_len) * len4 + k;

    let tw1_stride = n / (half_len * 2u);
    let tw2_stride = n / len4;

    let w1 = twiddles[k * tw1_stride];
    let w2 = twiddles[k * tw2_stride];
    let w3 = twiddles[(k + half_len) * tw2_stride];

    let x0 = data[pos];
    let x1 = data[pos + half_len];
    let x2 = data[pos + (2u * half_len)];
    let x3 = data[pos + (3u * half_len)];

    let x1w = mul_montgomery_u256(x1, w1);
    let x3w = mul_montgomery_u256(x3, w1);

    let a0 = add_fr(x0, x1w);
    let a1 = sub_fr(x0, x1w);
    let a2 = add_fr(x2, x3w);
    let a3 = sub_fr(x2, x3w);

    let a2w = mul_montgomery_u256(a2, w2);
    let a3w = mul_montgomery_u256(a3, w3);

    data[pos] = add_fr(a0, a2w);
    data[pos + half_len] = add_fr(a1, a3w);
    data[pos + (2u * half_len)] = sub_fr(a0, a2w);
    data[pos + (3u * half_len)] = sub_fr(a1, a3w);
}

// ============================================================================
// NTT TILE EXECUTION (Cooley-Tukey Radix-2)
// ============================================================================

@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn ntt_tile(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tile_offset = group_id.x * ELEMENTS_PER_TILE;
    let n_total = arrayLength(&twiddles);
    let n = min(ELEMENTS_PER_TILE, n_total - tile_offset);
    if n == 0u { return; }
    if n > ELEMENTS_PER_TILE { return; }

    var log2_elements = 0u;
    var m = n;
    while m > 1u {
        m = m >> 1u;
        log2_elements = log2_elements + 1u;
    }

    let local_idx_1 = local_id.x;
    let local_idx_2 = local_id.x + THREADS_PER_WORKGROUP;

    var load_idx_1 = local_idx_1;
    var load_idx_2 = local_idx_2;
    if n_total <= ELEMENTS_PER_TILE {
        load_idx_1 = reverse_bits(local_idx_1, log2_elements);
        load_idx_2 = reverse_bits(local_idx_2, log2_elements);
    }

    if local_idx_1 < n {
        shared_data[load_idx_1] = data[tile_offset + local_idx_1];
    }
    if local_idx_2 < n {
        shared_data[load_idx_2] = data[tile_offset + local_idx_2];
    }

    // The twiddles needed for all stages in this tile are subsets of the 
    // twiddles needed for the final stage. We load them exactly once.
    let twiddle_base_stride = n_total / n;
    if local_id.x < n / 2u {
        shared_twiddles[local_id.x] = twiddles[local_id.x * twiddle_base_stride];
    }

    workgroupBarrier();

    var half_len: u32 = 1u;
    for (var stage: u32 = 0u; stage < log2_elements; stage = stage + 1u) {
        let len = half_len * 2u;

        let butterfly_count = n / 2u;
        if local_id.x < butterfly_count {
            let k = local_id.x % half_len;
            let pos = (local_id.x / half_len) * len + k;

            // Fetch from shared memory using a localized stride!
            let shared_twiddle_stride = n / len;
            let twiddle = shared_twiddles[k * shared_twiddle_stride];

            let u = shared_data[pos];
            let v = shared_data[pos + half_len];

            let v_omega = mul_montgomery_u256(v, twiddle);
            shared_data[pos] = add_fr(u, v_omega);
            shared_data[pos + half_len] = sub_fr(u, v_omega);
        }

        half_len = len;
        workgroupBarrier();
    }

    if local_idx_1 < n {
        data[tile_offset + local_idx_1] = shared_data[local_idx_1];
    }
    if local_idx_2 < n {
        data[tile_offset + local_idx_2] = shared_data[local_idx_2];
    }
}
