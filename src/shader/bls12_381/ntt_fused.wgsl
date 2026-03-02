// src/shader/bls12_381/ntt_fused.wgsl
//
// Fused NTT + coset shift kernel. Identical to ntt_tile but multiplies each
// output element by a precomputed shift factor during write-back, eliminating
// a separate coset_shift dispatch.
//
// Concatenated after ntt.wgsl — reuses add_fr, sub_fr, reverse_bits,
// shared_data, THREADS_PER_WORKGROUP, ELEMENTS_PER_TILE, data, twiddles.
//
// Shift factors live in @group(1) to avoid conflicting with ntt.wgsl's
// @group(0) @binding(2) params uniform (used by other NTT entry points).

@group(1) @binding(0)
var<storage, read> shift_factors: array<U256>;

@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn ntt_tile_with_shift(
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

    workgroupBarrier();

    var half_len: u32 = 1u;
    for (var stage: u32 = 0u; stage < log2_elements; stage = stage + 1u) {
        let len = half_len * 2u;

        let butterfly_count = n / 2u;
        if local_id.x < butterfly_count {
            let k = local_id.x % half_len;
            let pos = (local_id.x / half_len) * len + k;

            let twiddle_stride = n_total / len;
            let twiddle = twiddles[k * twiddle_stride];

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
        let gi1 = tile_offset + local_idx_1;
        data[gi1] = mul_montgomery_u256(shared_data[local_idx_1], shift_factors[gi1]);
    }
    if local_idx_2 < n {
        let gi2 = tile_offset + local_idx_2;
        data[gi2] = mul_montgomery_u256(shared_data[local_idx_2], shift_factors[gi2]);
    }
}
