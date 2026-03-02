// src/shader/bls12_381/ntt_fused.wgsl
//
// Fused NTT + coset shift kernel. Identical to ntt_tile but multiplies each
// output element by a precomputed shift factor during write-back, eliminating
// a separate coset_shift dispatch.

@group(1) @binding(0)
var<storage, read> shift_factors: array<U256>;

fn ntt_tile_load_and_cache(
    local_id: vec3<u32>,
    group_id: vec3<u32>,
    apply_bitreverse_load: bool
) -> vec2<u32> {
    let tile_offset = group_id.x * ELEMENTS_PER_TILE;
    let n_total = arrayLength(&twiddles);
    let n = min(ELEMENTS_PER_TILE, n_total - tile_offset);
    if n == 0u { return vec2<u32>(0u, 0u); }

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
    if apply_bitreverse_load && n_total <= ELEMENTS_PER_TILE {
        load_idx_1 = reverse_bits(local_idx_1, log2_elements);
        load_idx_2 = reverse_bits(local_idx_2, log2_elements);
    }

    if local_idx_1 < n {
        shared_data[load_idx_1] = data[tile_offset + local_idx_1];
    }
    if local_idx_2 < n {
        shared_data[load_idx_2] = data[tile_offset + local_idx_2];
    }

    let twiddle_base_stride = n_total / n;
    if local_id.x < n / 2u {
        shared_twiddles[local_id.x] = twiddles[local_id.x * twiddle_base_stride];
    }

    return vec2<u32>(tile_offset, n);
}

fn ntt_tile_load_pointwise_and_cache(
    local_id: vec3<u32>,
    group_id: vec3<u32>
) -> vec2<u32> {
    let tile_offset = group_id.x * ELEMENTS_PER_TILE;
    let n_total = arrayLength(&twiddles);
    let n = min(ELEMENTS_PER_TILE, n_total - tile_offset);
    if n == 0u { return vec2<u32>(0u, 0u); }

    let local_idx_1 = local_id.x;
    let local_idx_2 = local_id.x + THREADS_PER_WORKGROUP;

    var log2_elements = 0u;
    var m = n;
    while m > 1u {
        m = m >> 1u;
        log2_elements = log2_elements + 1u;
    }

    var load_idx_1 = local_idx_1;
    var load_idx_2 = local_idx_2;
    if n_total <= ELEMENTS_PER_TILE {
        load_idx_1 = reverse_bits(local_idx_1, log2_elements);
        load_idx_2 = reverse_bits(local_idx_2, log2_elements);
    }

    if local_idx_1 < n {
        let gi1 = tile_offset + local_idx_1;
        let ab1 = mul_montgomery_u256(pointwise_a[gi1], pointwise_b[gi1]);
        let ab_c1 = sub_fr(ab1, pointwise_c[gi1]);
        shared_data[load_idx_1] = mul_montgomery_u256(ab_c1, pointwise_z_inv[0u]);
    }
    if local_idx_2 < n {
        let gi2 = tile_offset + local_idx_2;
        let ab2 = mul_montgomery_u256(pointwise_a[gi2], pointwise_b[gi2]);
        let ab_c2 = sub_fr(ab2, pointwise_c[gi2]);
        shared_data[load_idx_2] = mul_montgomery_u256(ab_c2, pointwise_z_inv[0u]);
    }

    let twiddle_base_stride = n_total / n;
    if local_id.x < n / 2u {
        shared_twiddles[local_id.x] = twiddles[local_id.x * twiddle_base_stride];
    }

    return vec2<u32>(tile_offset, n);
}

fn run_ntt_tile_dit(local_id: vec3<u32>, n: u32) {
    var log2_elements = 0u;
    var m = n;
    while m > 1u {
        m = m >> 1u;
        log2_elements = log2_elements + 1u;
    }

    workgroupBarrier();
    var half_len: u32 = 1u;
    for (var stage: u32 = 0u; stage < log2_elements; stage = stage + 1u) {
        let len = half_len * 2u;
        let butterfly_count = n / 2u;
        if local_id.x < butterfly_count {
            let k = local_id.x % half_len;
            let pos = (local_id.x / half_len) * len + k;
            let twiddle = shared_twiddles[k * (n / len)];
            let u = shared_data[pos];
            let v = shared_data[pos + half_len];
            let v_omega = mul_montgomery_u256(v, twiddle);
            shared_data[pos] = add_fr(u, v_omega);
            shared_data[pos + half_len] = sub_fr(u, v_omega);
        }
        half_len = len;
        workgroupBarrier();
    }
}

fn run_ntt_tile_dif(local_id: vec3<u32>, n: u32) {
    workgroupBarrier();
    var half_len = n / 2u;
    loop {
        if half_len == 0u {
            break;
        }
        let len = half_len * 2u;
        let butterfly_count = n / 2u;
        if local_id.x < butterfly_count {
            let k = local_id.x % half_len;
            let pos = (local_id.x / half_len) * len + k;
            let twiddle = shared_twiddles[k * (n / len)];
            let u = shared_data[pos];
            let v = shared_data[pos + half_len];
            let sum = add_fr(u, v);
            let diff = sub_fr(u, v);
            shared_data[pos] = sum;
            shared_data[pos + half_len] = mul_montgomery_u256(diff, twiddle);
        }
        workgroupBarrier();
        half_len = half_len >> 1u;
    }
}

fn writeback_tile_with_optional_shift(
    local_id: vec3<u32>,
    tile_offset: u32,
    n: u32,
    apply_shift: bool
) {
    let local_idx_1 = local_id.x;
    let local_idx_2 = local_id.x + THREADS_PER_WORKGROUP;

    if local_idx_1 < n {
        let gi1 = tile_offset + local_idx_1;
        let val1 = shared_data[local_idx_1];
        if apply_shift {
            data[gi1] = mul_montgomery_u256(val1, shift_factors[gi1]);
        } else {
            data[gi1] = val1;
        }
    }
    if local_idx_2 < n {
        let gi2 = tile_offset + local_idx_2;
        let val2 = shared_data[local_idx_2];
        if apply_shift {
            data[gi2] = mul_montgomery_u256(val2, shift_factors[gi2]);
        } else {
            data[gi2] = val2;
        }
    }
}

@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn ntt_tile_with_shift(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tile = ntt_tile_load_and_cache(local_id, group_id, true);
    let tile_offset = tile.x;
    let n = tile.y;
    if n == 0u { return; }

    run_ntt_tile_dit(local_id, n);
    writeback_tile_with_optional_shift(local_id, tile_offset, n, true);
}

@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn ntt_tile_dit_no_bitreverse(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tile = ntt_tile_load_and_cache(local_id, group_id, false);
    let tile_offset = tile.x;
    let n = tile.y;
    if n == 0u { return; }

    run_ntt_tile_dit(local_id, n);
    writeback_tile_with_optional_shift(local_id, tile_offset, n, false);
}

@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn ntt_tile_dif(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tile = ntt_tile_load_and_cache(local_id, group_id, false);
    let tile_offset = tile.x;
    let n = tile.y;
    if n == 0u { return; }

    run_ntt_tile_dif(local_id, n);
    writeback_tile_with_optional_shift(local_id, tile_offset, n, false);
}

@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn ntt_tile_dif_with_shift(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tile = ntt_tile_load_and_cache(local_id, group_id, false);
    let tile_offset = tile.x;
    let n = tile.y;
    if n == 0u { return; }

    run_ntt_tile_dif(local_id, n);
    writeback_tile_with_optional_shift(local_id, tile_offset, n, true);
}

@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn ntt_tile_fused_pointwise(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tile = ntt_tile_load_pointwise_and_cache(local_id, group_id);
    let tile_offset = tile.x;
    let n = tile.y;
    if n == 0u { return; }

    run_ntt_tile_dit(local_id, n);
    writeback_tile_with_optional_shift(local_id, tile_offset, n, true);
}
