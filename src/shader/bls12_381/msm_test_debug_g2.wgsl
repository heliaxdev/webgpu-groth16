// src/shader/bls12_381/msm_test_debug_g2.wgsl

@group(0) @binding(0) var<storage, read> rt_in_g2: array<PointG2>;
@group(0) @binding(1) var<storage, read_write> rt_out_g2: array<PointG2>;

@compute @workgroup_size(64)
fn roundtrip_g2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&rt_in_g2) { return; }
    rt_out_g2[i] = store_g2(load_g2(rt_in_g2[i]));
}

@group(0) @binding(0) var<storage, read> rt_add_g2_in_a: array<PointG2>;
@group(0) @binding(1) var<storage, read> rt_add_g2_in_b: array<PointG2>;
@group(0) @binding(2) var<storage, read_write> rt_add_g2_out: array<PointG2>;

@compute @workgroup_size(1)
fn roundtrip_add_g2_complete(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&rt_add_g2_in_a) { return; }
    let a = load_g2(rt_add_g2_in_a[i]);
    let b = load_g2(rt_add_g2_in_b[i]);
    let sum = add_g2_complete(a, b);
    rt_add_g2_out[i] = store_g2_proj(sum);
}
