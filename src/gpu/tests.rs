use std::borrow::Cow;

use blstrs::{Bls12, G1Affine, Scalar};
use ff::Field;
use group::prime::PrimeCurveAffine;

use super::*;
use crate::gpu::curve::GpuCurve;

/// Dispatches a single-workgroup compute shader test.
///
/// Creates a pipeline from the MSM shader source with the given `entry_point`,
/// binds the provided `buffers` according to `buf_kinds`, dispatches (1,1,1),
/// and submits.
fn dispatch_shader_test(
    gpu: &GpuContext<Bls12>,
    source: &str,
    entry_point: &str,
    buf_kinds: &[BufKind],
    buffers: &[&wgpu::Buffer],
) {
    let shader =
        gpu.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("test shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
            });

    let bgl = create_bind_group_layout(&gpu.device, "test bgl", buf_kinds);
    let layout = pipeline_layout(&gpu.device, &[&bgl]);
    let pipeline = create_pipeline(
        &gpu.device,
        "test pipeline",
        &layout,
        &shader,
        entry_point,
    );

    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|(i, buf)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buf.as_entire_binding(),
        })
        .collect();

    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test bg"),
        layout: &bgl,
        entries: &entries,
    });

    let mut encoder =
        gpu.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test encoder"),
            });
    {
        let mut pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("test pass"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

#[tokio::test]
async fn test_g1_cpu_gpu_cpu_roundtrip_bytes_and_deserialize() {
    let gpu = GpuContext::<Bls12>::new()
        .await
        .expect("failed to init gpu context");

    let point = G1Affine::generator();
    let bytes = <Bls12 as GpuCurve>::serialize_g1(&point);
    let buf = gpu.create_storage_buffer("g1_roundtrip", &bytes);
    let read_back = gpu
        .read_buffer(&buf, bytes.len() as u64)
        .await
        .expect("failed to read back g1 bytes");

    assert_eq!(bytes, read_back, "raw gpu roundtrip bytes differ");

    let parsed = <Bls12 as GpuCurve>::deserialize_g1(&read_back)
        .expect("deserializing round-tripped g1 bytes failed");
    let parsed_affine: G1Affine = parsed.into();
    assert_eq!(parsed_affine, point, "g1 roundtrip point mismatch");
}

#[tokio::test]
async fn test_g1_shader_load_store_roundtrip() {
    let gpu = GpuContext::<Bls12>::new()
        .await
        .expect("failed to init gpu context");

    let point = G1Affine::generator();
    let in_bytes = <Bls12 as GpuCurve>::serialize_g1(&point);

    let in_buf = gpu.create_storage_buffer("rt_in_g1", &in_bytes);
    let out_buf = gpu.create_empty_buffer("rt_out_g1", in_bytes.len() as u64);

    dispatch_shader_test(
        &gpu,
        <Bls12 as GpuCurve>::TEST_SHADER_G1_SOURCE,
        "roundtrip_g1",
        &[BufKind::ReadOnly, BufKind::ReadWrite],
        &[&in_buf, &out_buf],
    );

    let out_bytes = gpu
        .read_buffer(&out_buf, in_bytes.len() as u64)
        .await
        .expect("failed to read rt g1 bytes");

    let parsed = <Bls12 as GpuCurve>::deserialize_g1(&out_bytes)
        .expect("deserializing shader round-tripped g1 bytes failed");
    let parsed_affine: G1Affine = parsed.into();
    assert_eq!(parsed_affine, point, "g1 shader roundtrip point mismatch");
}

#[tokio::test]
async fn test_g1_shader_coord_only_montgomery_roundtrip() {
    let gpu = GpuContext::<Bls12>::new()
        .await
        .expect("failed to init gpu context");

    let point = G1Affine::generator();
    let in_bytes = <Bls12 as GpuCurve>::serialize_g1(&point);

    let in_buf = gpu.create_storage_buffer("rt_in_coords_g1", &in_bytes);
    let out_buf =
        gpu.create_empty_buffer("rt_out_coords_g1", in_bytes.len() as u64);

    dispatch_shader_test(
        &gpu,
        <Bls12 as GpuCurve>::TEST_SHADER_G1_SOURCE,
        "roundtrip_coords_g1",
        &[BufKind::ReadOnly, BufKind::ReadWrite],
        &[&in_buf, &out_buf],
    );

    let out_bytes = gpu
        .read_buffer(&out_buf, in_bytes.len() as u64)
        .await
        .expect("failed to read rt coords g1 bytes");

    let parsed = <Bls12 as GpuCurve>::deserialize_g1(&out_bytes)
        .expect("deserializing coord round-tripped g1 bytes failed");
    let parsed_affine: G1Affine = parsed.into();
    assert_eq!(parsed_affine, point, "g1 coord roundtrip point mismatch");
}

#[tokio::test]
async fn test_scalar_to_from_montgomery_roundtrip() {
    let gpu = GpuContext::<Bls12>::new()
        .await
        .expect("failed to init gpu context");

    let scalars = vec![
        Scalar::ZERO,
        Scalar::ONE,
        Scalar::from(2u64),
        Scalar::from(3u64),
        Scalar::from(0x1234_5678_9abc_def0u64),
        -Scalar::from(5u64),
    ];

    let mut bytes = Vec::with_capacity(scalars.len() * 32);
    for s in &scalars {
        bytes.extend_from_slice(&<Bls12 as GpuCurve>::serialize_scalar(s));
    }

    let buf = gpu.create_storage_buffer("scalar_roundtrip", &bytes);
    gpu.execute_to_montgomery(&buf, scalars.len() as u32);
    gpu.execute_from_montgomery(&buf, scalars.len() as u32);
    let out = gpu
        .read_buffer(&buf, bytes.len() as u64)
        .await
        .expect("failed to read scalar roundtrip");

    for (i, chunk) in out.chunks_exact(32).enumerate() {
        let got = <Bls12 as GpuCurve>::deserialize_scalar(chunk)
            .expect("deserialize scalar failed");
        assert_eq!(got, scalars[i], "scalar mismatch at index {i}");
    }
}

#[tokio::test]
async fn test_g1_shader_double_roundtrip() {
    use group::Curve;

    let gpu = GpuContext::<Bls12>::new()
        .await
        .expect("failed to init gpu context");

    let point = G1Affine::generator();
    let in_bytes = <Bls12 as GpuCurve>::serialize_g1(&point);

    let in_buf = gpu.create_storage_buffer("rt_in_g1", &in_bytes);
    let out_buf = gpu.create_empty_buffer("rt_out_g1", in_bytes.len() as u64);

    dispatch_shader_test(
        &gpu,
        <Bls12 as GpuCurve>::TEST_SHADER_G1_SOURCE,
        "roundtrip_double_g1",
        &[BufKind::ReadOnly, BufKind::ReadWrite],
        &[&in_buf, &out_buf],
    );

    let out_bytes = gpu
        .read_buffer(&out_buf, in_bytes.len() as u64)
        .await
        .expect("failed to read rt double g1 bytes");

    let g_proj: blstrs::G1Projective = point.into();
    let expected: G1Affine = (g_proj + g_proj).to_affine();

    let parsed = <Bls12 as GpuCurve>::deserialize_g1(&out_bytes)
        .expect("GPU double_g1 produced invalid curve point");
    let gpu_affine: G1Affine = parsed.into();
    assert_eq!(gpu_affine, expected, "GPU double_g1 mismatch");
}

#[tokio::test]
async fn test_g2_add_complete_roundtrip() {
    use blstrs::{G2Affine, G2Projective};
    use group::Curve;

    let gpu = GpuContext::<Bls12>::new()
        .await
        .expect("failed to init gpu context");

    let generator = G2Affine::generator();
    let g_proj: G2Projective = generator.into();
    let three_g: G2Affine = (g_proj + g_proj + g_proj).to_affine();

    // Test 1: G + 3G = 4G (distinct points)
    let a_bytes = <Bls12 as GpuCurve>::serialize_g2(&generator);
    let b_bytes = <Bls12 as GpuCurve>::serialize_g2(&three_g);

    let a_buf = gpu.create_storage_buffer("rt_add_g2_a", &a_bytes);
    let b_buf = gpu.create_storage_buffer("rt_add_g2_b", &b_bytes);
    let out_buf =
        gpu.create_empty_buffer("rt_add_g2_out", a_bytes.len() as u64);

    dispatch_shader_test(
        &gpu,
        <Bls12 as GpuCurve>::TEST_SHADER_G2_SOURCE,
        "roundtrip_add_g2_complete",
        &[BufKind::ReadOnly, BufKind::ReadOnly, BufKind::ReadWrite],
        &[&a_buf, &b_buf, &out_buf],
    );

    let out_bytes = gpu
        .read_buffer(&out_buf, a_bytes.len() as u64)
        .await
        .expect("failed to read rt add g2 bytes");

    let expected: G2Affine = (g_proj + g_proj + g_proj + g_proj).to_affine();
    let parsed = <Bls12 as GpuCurve>::deserialize_g2(&out_bytes)
        .expect("GPU add_g2_complete produced invalid curve point");
    let gpu_affine: G2Affine = parsed.into();
    assert_eq!(gpu_affine, expected, "GPU add_g2_complete G+3G mismatch");

    // Test 2: G + G = 2G (doubling via complete formula)
    let b_buf_2 = gpu.create_storage_buffer("rt_add_g2_b2", &a_bytes);
    let out_buf_2 =
        gpu.create_empty_buffer("rt_add_g2_out2", a_bytes.len() as u64);

    dispatch_shader_test(
        &gpu,
        <Bls12 as GpuCurve>::TEST_SHADER_G2_SOURCE,
        "roundtrip_add_g2_complete",
        &[BufKind::ReadOnly, BufKind::ReadOnly, BufKind::ReadWrite],
        &[&a_buf, &b_buf_2, &out_buf_2],
    );

    let out_bytes_2 = gpu
        .read_buffer(&out_buf_2, a_bytes.len() as u64)
        .await
        .expect("failed to read rt add g2 doubling bytes");

    let expected_double: G2Affine = (g_proj + g_proj).to_affine();
    let parsed_double = <Bls12 as GpuCurve>::deserialize_g2(&out_bytes_2)
        .expect("GPU add_g2_complete doubling produced invalid curve point");
    let gpu_affine_double: G2Affine = parsed_double.into();
    assert_eq!(
        gpu_affine_double, expected_double,
        "GPU add_g2_complete G+G (doubling) mismatch"
    );
}

/// Test: parallel tree reduction of 64 G1 points in var<workgroup> memory.
///
/// Exercises `var<workgroup> shared: array<PointG1, 64>` with
/// `@workgroup_size(64)` to diagnose whether Metal threadgroup memory works
/// correctly with 360-byte PointG1 structs.
#[tokio::test]
async fn test_g1_workgroup_tree_reduction() {
    use group::{Curve, Group};

    let gpu = GpuContext::<Bls12>::new()
        .await
        .expect("failed to init gpu context");

    let generator = G1Affine::generator();
    let gen_proj: blstrs::G1Projective = generator.into();

    // Generate 64 distinct points: i*G for i=1..=64
    let mut points = Vec::with_capacity(64);
    let mut running = gen_proj;
    for _ in 0..64 {
        points.push(running.to_affine());
        running += gen_proj;
    }

    // Compute expected sum on CPU: sum(i*G, i=1..64) = (64*65/2)*G = 2080*G
    let mut cpu_sum = blstrs::G1Projective::identity();
    for p in &points {
        let proj: blstrs::G1Projective = (*p).into();
        cpu_sum += proj;
    }
    let expected: G1Affine = cpu_sum.to_affine();

    // Serialize 64 points to GPU format
    let mut in_bytes =
        Vec::with_capacity(64 * <Bls12 as GpuCurve>::G1_GPU_BYTES);
    for p in &points {
        in_bytes.extend_from_slice(&<Bls12 as GpuCurve>::serialize_g1(p));
    }

    let in_buf = gpu.create_storage_buffer("wg_test_in_g1", &in_bytes);
    let out_buf = gpu.create_empty_buffer(
        "wg_test_out_g1",
        <Bls12 as GpuCurve>::G1_GPU_BYTES as u64,
    );

    dispatch_shader_test(
        &gpu,
        <Bls12 as GpuCurve>::TEST_SHADER_G1_SOURCE,
        "test_workgroup_reduction_g1",
        &[BufKind::ReadOnly, BufKind::ReadWrite],
        &[&in_buf, &out_buf],
    );

    let out_bytes = gpu
        .read_buffer(&out_buf, <Bls12 as GpuCurve>::G1_GPU_BYTES as u64)
        .await
        .expect("failed to read workgroup reduction output");

    let parsed = <Bls12 as GpuCurve>::deserialize_g1(&out_bytes)
        .expect("GPU workgroup tree reduction produced invalid curve point");
    let gpu_affine: G1Affine = parsed.into();
    assert_eq!(
        gpu_affine, expected,
        "GPU workgroup tree reduction mismatch: sum of i*G for i=1..64 should \
         be 2080*G"
    );
}
