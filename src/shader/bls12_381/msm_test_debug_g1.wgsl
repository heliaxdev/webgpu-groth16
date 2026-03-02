// src/shader/bls12_381/msm_test_debug_g1.wgsl

@group(0) @binding(0) var<storage, read> rt_in_g1: array<PointG1>;
@group(0) @binding(1) var<storage, read_write> rt_out_g1: array<PointG1>;

@compute @workgroup_size(64)
fn roundtrip_g1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&rt_in_g1) { return; }
    rt_out_g1[i] = store_g1(load_g1(rt_in_g1[i]));
}

@compute @workgroup_size(1)
fn roundtrip_double_g1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&rt_in_g1) { return; }
    let p = load_g1(rt_in_g1[i]);
    let doubled = double_g1(p);
    rt_out_g1[i] = store_g1(doubled);
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

@group(0) @binding(0) var<storage, read> wg_test_in_g1: array<PointG1>;
@group(0) @binding(1) var<storage, read_write> wg_test_out_g1: array<PointG1>;

var<workgroup> wg_test_shared: array<PointG1, 64>;

@compute @workgroup_size(64)
fn test_workgroup_reduction_g1(
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;

    wg_test_shared[tid] = load_g1(wg_test_in_g1[tid]);
    workgroupBarrier();

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
