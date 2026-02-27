// src/shader/bls12_381/ntt.wgsl

// ============================================================================
// CONSTANTS & WORKGROUP SIZE
// ============================================================================
// We use a workgroup size of 256 threads.
// Each thread processes 2 elements, so one workgroup handles up to 512 elements.
const THREADS_PER_WORKGROUP: u32 = 256u;
const ELEMENTS_PER_TILE: u32 = 512u;

// ============================================================================
// GPU PIPELINE BUFFERS
// ============================================================================

@group(0) @binding(0)
var<storage, read_write> data: array<U256>; // The polynomial coefficients / evaluations

@group(0) @binding(1)
var<storage, read> twiddles: array<U256>; // Precomputed powers of omega in Montgomery form

// Fast workgroup-shared memory for the Cooley-Tukey butterflies
var<workgroup> shared_data: array<U256, ELEMENTS_PER_TILE>;

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
    var result: u32 = 0u;
    var temp: u32 = n;
    for (var i: u32 = 0u; i < bit_len; i = i + 1u) {
        result = (result << 1u) | (temp & 1u);
        temp = temp >> 1u;
    }
    return result;
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

    // This shader implements a single-tile NTT. Larger domains are not yet
    // supported by this kernel and are left untouched for out-of-range groups.
    if n > ELEMENTS_PER_TILE { return; }
    
    // 1. Bit-Reversed Load from Global VRAM to Workgroup Memory
    // The Cooley-Tukey algorithm yields elements out of order, requiring bit-inversion.
    // We apply the bit-reversal permutation while loading into shared memory.
    var log2_elements = 0u;
    var m = n;
    while m > 1u {
        m = m >> 1u;
        log2_elements = log2_elements + 1u;
    }

    let local_idx_1 = local_id.x;
    let local_idx_2 = local_id.x + THREADS_PER_WORKGROUP;

    let rev_idx_1 = reverse_bits(local_idx_1, log2_elements);
    let rev_idx_2 = reverse_bits(local_idx_2, log2_elements);

    if local_idx_1 < n {
        shared_data[rev_idx_1] = data[tile_offset + local_idx_1];
    }
    if local_idx_2 < n {
        shared_data[rev_idx_2] = data[tile_offset + local_idx_2];
    }

    workgroupBarrier();
    
    // 2. Cooley-Tukey Radix-2 In-Place Butterfly Iterations
    // Iterates through log2(N) stages.
    var half_len: u32 = 1u;
    for (var stage: u32 = 0u; stage < log2_elements; stage = stage + 1u) {
        let len = half_len * 2u;
        
        let butterfly_count = n / 2u;
        if local_id.x < butterfly_count {
            // Each thread handles one butterfly operation.
            let k = local_id.x % half_len;
            let pos = (local_id.x / half_len) * len + k;

            let twiddle_stride = n / len;
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
    
    // 3. Store the processed tile back to Global VRAM
    if local_idx_1 < n {
        data[tile_offset + local_idx_1] = shared_data[local_idx_1];
    }
    if local_idx_2 < n {
        data[tile_offset + local_idx_2] = shared_data[local_idx_2];
    }
}
