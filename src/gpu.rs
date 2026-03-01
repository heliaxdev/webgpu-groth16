//! GPU context, compute pipeline management, and kernel dispatch.
//!
//! [`GpuContext`] owns the wgpu device/queue and all pre-compiled compute pipelines
//! needed for MSM, NTT, and polynomial operations. Submodules provide dispatch
//! methods as `impl GpuContext` blocks:
//!
//! - [`msm`] — MSM 5-kernel Pippenger pipeline (to_montgomery, aggregate, reduce,
//!   weight, subsum)
//! - [`ntt`] — NTT dispatchers (tile-local and multi-stage global), Montgomery
//!   conversion, coset shift, pointwise polynomial evaluation
//! - [`h_poly`] — H-polynomial pipeline (fused NTT+shift → pointwise → iNTT)
//! - [`curve`] — CPU↔GPU serialization bridge for BLS12-381 curve elements

mod buffers;
pub mod curve;
mod h_poly;
mod msm;
mod ntt;

use std::borrow::Cow;
use std::marker::PhantomData;
#[cfg(feature = "profiling")]
use std::sync::Mutex;
#[cfg(feature = "timing")]
use std::time::Instant;

use anyhow::Context;
use wgpu::util::DeviceExt;

use self::curve::GpuCurve;

/// Workgroup size for scalar kernels (Montgomery, coset shift, pointwise, NTT global stage).
/// Matches `@workgroup_size(256)` in the corresponding WGSL shaders.
const SCALAR_WORKGROUP_SIZE: u32 = 256;

/// Number of elements processed per NTT tile workgroup (256 threads x 2 elements).
/// Also the threshold: NTT sizes > this use the multi-stage global algorithm.
const NTT_TILE_SIZE: u32 = 512;

/// Workgroup size for MSM kernels (aggregate, weight, to_montgomery_bases).
/// Matches `@workgroup_size(64)` in msm.wgsl.
const MSM_WORKGROUP_SIZE: u32 = 64;

/// G1 subsum uses single-pass parallel shared-memory tree reduction:
/// one workgroup of 64 threads per window. chunks_per_window=1 since Phase 1
/// produces the final window sum directly.
const G1_SUBSUM_CHUNKS_PER_WINDOW: u32 = 1;

/// Number of workgroup chunks per window in G2 two-phase subsum reduction.
const G2_SUBSUM_CHUNKS_PER_WINDOW: u32 = 32;

/// Creates a compute pass, optionally wrapped in a GPU profiling scope.
///
/// With the `profiling` feature enabled, wraps the pass in
/// `scope.scoped_compute_pass()` for per-kernel GPU timing via
/// [`wgpu_profiler`]. Without it, creates a plain compute pass.
///
/// Usage: `let mut cpass = compute_pass!(scope, encoder, "kernel_label");`
macro_rules! compute_pass {
    ($scope:expr, $encoder:expr, $label:expr) => {{
        #[cfg(feature = "profiling")]
        let pass = $scope.scoped_compute_pass($label);
        #[cfg(not(feature = "profiling"))]
        let pass = $encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some($label),
            timestamp_writes: None,
        });
        pass
    }};
}
pub(crate) use compute_pass;

/// Shorthand for buffer binding types in compute bind group layouts.
enum BufKind {
    ReadOnly,
    ReadWrite,
    Uniform,
}

/// Creates a compute bind group layout with sequentially-numbered bindings.
fn create_bind_group_layout(
    device: &wgpu::Device,
    label: &str,
    bindings: &[BufKind],
) -> wgpu::BindGroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
        .iter()
        .enumerate()
        .map(|(i, kind)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: match kind {
                BufKind::ReadOnly => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                BufKind::ReadWrite => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                BufKind::Uniform => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

/// Creates a pipeline layout from one or more bind group layouts.
fn pipeline_layout(
    device: &wgpu::Device,
    bind_group_layouts: &[&wgpu::BindGroupLayout],
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts,
        immediate_size: 0,
    })
}

/// Creates a compute pipeline with the given layout, shader module, and entry point.
fn create_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// GPU buffers required for an MSM dispatch.
pub struct MsmBuffers<'a> {
    pub bases: &'a wgpu::Buffer,
    pub base_indices: &'a wgpu::Buffer,
    pub bucket_pointers: &'a wgpu::Buffer,
    pub bucket_sizes: &'a wgpu::Buffer,
    pub aggregated_buckets: &'a wgpu::Buffer,
    pub bucket_values: &'a wgpu::Buffer,
    pub window_starts: &'a wgpu::Buffer,
    pub window_counts: &'a wgpu::Buffer,
    pub window_sums: &'a wgpu::Buffer,
    /// Sub-bucket reduce buffers (only used when has_chunks is true).
    pub reduce_starts: Option<&'a wgpu::Buffer>,
    pub reduce_counts: Option<&'a wgpu::Buffer>,
    /// Original (pre-chunking) bucket values for weight pass.
    pub orig_bucket_values: Option<&'a wgpu::Buffer>,
    /// Original window starts/counts for subsum passes.
    pub orig_window_starts: Option<&'a wgpu::Buffer>,
    pub orig_window_counts: Option<&'a wgpu::Buffer>,
}

/// GPU buffers required for the H polynomial pipeline.
pub struct HPolyBuffers<'a> {
    pub a: &'a wgpu::Buffer,
    pub b: &'a wgpu::Buffer,
    pub c: &'a wgpu::Buffer,
    pub h: &'a wgpu::Buffer,
    pub twiddles_inv: &'a wgpu::Buffer,
    pub twiddles_fwd: &'a wgpu::Buffer,
    pub shifts: &'a wgpu::Buffer,
    pub inv_shifts: &'a wgpu::Buffer,
    pub z_invs: &'a wgpu::Buffer,
}

pub struct GpuContext<C> {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    // Polynomial Pipelines
    pub ntt_pipeline: wgpu::ComputePipeline,
    pub ntt_fused_pipeline: wgpu::ComputePipeline,
    pub ntt_global_stage_pipeline: wgpu::ComputePipeline,
    pub ntt_bitreverse_pipeline: wgpu::ComputePipeline,
    pub coset_shift_pipeline: wgpu::ComputePipeline,
    pub pointwise_poly_pipeline: wgpu::ComputePipeline,
    pub to_montgomery_pipeline: wgpu::ComputePipeline,
    pub from_montgomery_pipeline: wgpu::ComputePipeline,

    // MSM 2-Stage Pipelines
    pub msm_agg_g1_pipeline: wgpu::ComputePipeline,
    pub msm_sum_g1_pipeline: wgpu::ComputePipeline,
    pub msm_agg_g2_pipeline: wgpu::ComputePipeline,
    pub msm_sum_g2_pipeline: wgpu::ComputePipeline,
    pub msm_to_mont_g1_pipeline: wgpu::ComputePipeline,
    pub msm_to_mont_g2_pipeline: wgpu::ComputePipeline,
    pub msm_weight_g1_pipeline: wgpu::ComputePipeline,
    pub msm_subsum_phase1_g1_pipeline: wgpu::ComputePipeline,
    pub msm_subsum_phase2_g1_pipeline: wgpu::ComputePipeline,
    pub msm_weight_g2_pipeline: wgpu::ComputePipeline,
    pub msm_subsum_phase1_g2_pipeline: wgpu::ComputePipeline,
    pub msm_subsum_phase2_g2_pipeline: wgpu::ComputePipeline,
    pub msm_reduce_g1_pipeline: wgpu::ComputePipeline,
    pub msm_reduce_g2_pipeline: wgpu::ComputePipeline,

    // Bind Group Layouts
    pub ntt_bind_group_layout: wgpu::BindGroupLayout,
    pub ntt_fused_shift_bgl: wgpu::BindGroupLayout,
    pub ntt_params_bind_group_layout: wgpu::BindGroupLayout,
    pub coset_shift_bind_group_layout: wgpu::BindGroupLayout,
    pub pointwise_poly_bind_group_layout: wgpu::BindGroupLayout,
    pub montgomery_bind_group_layout: wgpu::BindGroupLayout,
    pub msm_agg_bind_group_layout: wgpu::BindGroupLayout,
    pub msm_sum_bind_group_layout: wgpu::BindGroupLayout,
    pub msm_weight_g1_bind_group_layout: wgpu::BindGroupLayout,
    pub msm_weight_g2_bind_group_layout: wgpu::BindGroupLayout,
    pub msm_subsum_phase1_bind_group_layout: wgpu::BindGroupLayout,
    pub msm_subsum_phase2_bind_group_layout: wgpu::BindGroupLayout,
    pub msm_reduce_bind_group_layout: wgpu::BindGroupLayout,

    _marker: PhantomData<C>,

    #[cfg(feature = "profiling")]
    pub profiler: Mutex<wgpu_profiler::GpuProfiler>,
}

impl<C: GpuCurve> GpuContext<C> {
    pub async fn new() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .context("Failed to find a compatible WebGPU adapter")?;

        #[cfg(feature = "profiling")]
        let required_features =
            adapter.features() & wgpu_profiler::GpuProfiler::ALL_WGPU_TIMER_FEATURES;
        #[cfg(not(feature = "profiling"))]
        let required_features = wgpu::Features::empty();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Groth16 Prover Device"),
                // Use adapter.limits() directly to support large buffers (>128MB) for WebAssembly
                required_limits: adapter.limits(),
                required_features,
                ..Default::default()
            })
            .await
            .context("Failed to request WebGPU device")?;

        macro_rules! timed {
            ($label:expr, $expr:expr) => {{
                #[cfg(feature = "timing")]
                let _t = Instant::now();
                let _r = $expr;
                #[cfg(feature = "timing")]
                eprintln!(
                    "[init] {:<30} {:>8.1}ms",
                    $label,
                    _t.elapsed().as_secs_f64() * 1000.0
                );
                _r
            }};
        }

        #[cfg(feature = "timing")]
        let init_start = Instant::now();

        // 1. Compile Shader Modules
        #[cfg(feature = "timing")]
        let shader_start = Instant::now();
        let ntt_module = timed!(
            "shader: NTT",
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("NTT Shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(C::NTT_SOURCE)),
            })
        );

        let msm_module = timed!(
            "shader: MSM",
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MSM Shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(C::MSM_SOURCE)),
            })
        );

        let poly_ops_module = timed!(
            "shader: Poly Ops",
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Poly Ops Shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(C::POLY_OPS_SOURCE)),
            })
        );

        let ntt_fused_module = timed!(
            "shader: NTT Fused",
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("NTT Fused Shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(C::NTT_FUSED_SOURCE)),
            })
        );
        #[cfg(feature = "timing")]
        let shader_total = shader_start.elapsed();

        // 2. Define Bind Group Layouts
        #[cfg(feature = "timing")]
        let layouts_start = Instant::now();
        use BufKind::{ReadOnly as RO, ReadWrite as RW, Uniform as UF};

        let ntt_bind_group_layout = create_bind_group_layout(&device, "NTT", &[RW, RO]);
        let ntt_fused_shift_bgl = create_bind_group_layout(&device, "NTT Fused Shift", &[RO]);
        let ntt_params_bind_group_layout =
            create_bind_group_layout(&device, "NTT Global", &[RW, RO, UF]);
        let coset_shift_bind_group_layout =
            create_bind_group_layout(&device, "Coset Shift", &[RW, RO]);
        let pointwise_poly_bind_group_layout =
            create_bind_group_layout(&device, "Pointwise Poly", &[RO, RO, RO, RW, RO]);
        let montgomery_bind_group_layout = create_bind_group_layout(&device, "Montgomery", &[RW]);
        let msm_agg_bind_group_layout =
            create_bind_group_layout(&device, "MSM Agg", &[RO, RO, RO, RO, RW, RO]);
        let msm_sum_bind_group_layout =
            create_bind_group_layout(&device, "MSM Sum", &[RO, RO, RO, RO, RW]);
        // Weight buckets: [data(rw), bucket_values(read)]
        let msm_weight_g1_bind_group_layout =
            create_bind_group_layout(&device, "MSM Weight G1", &[RW, RO]);
        let msm_weight_g2_bind_group_layout =
            create_bind_group_layout(&device, "MSM Weight G2", &[RW, RO]);
        // Phase1: [agg_buckets(read), window_starts(read), window_counts(read),
        //          partial_sums(rw), subsum_params(uniform)]
        let msm_subsum_phase1_bind_group_layout =
            create_bind_group_layout(&device, "MSM Subsum Phase1", &[RO, RO, RO, RW, UF]);
        // Phase2: [partial_sums(read), window_sums(rw), subsum_params(uniform)]
        let msm_subsum_phase2_bind_group_layout =
            create_bind_group_layout(&device, "MSM Subsum Phase2", &[RO, RW, UF]);
        // Reduce sub-buckets: [input(read), starts(read), counts(read), output(rw)]
        let msm_reduce_bind_group_layout =
            create_bind_group_layout(&device, "MSM Reduce", &[RO, RO, RO, RW]);

        #[cfg(feature = "timing")]
        let layouts_total = layouts_start.elapsed();
        #[cfg(feature = "timing")]
        eprintln!(
            "[init] {:<30} {:>8.1}ms",
            "bind group layouts (total)",
            layouts_total.as_secs_f64() * 1000.0
        );

        // 3. Create Compute Pipelines
        #[cfg(feature = "timing")]
        let pipelines_start = Instant::now();

        // NTT pipelines
        let ntt_tile_layout = pipeline_layout(&device, &[&ntt_bind_group_layout]);
        let ntt_global_layout = pipeline_layout(&device, &[&ntt_params_bind_group_layout]);
        let ntt_pipeline = timed!(
            "pipeline: NTT Tile",
            create_pipeline(
                &device,
                "NTT Tile",
                &ntt_tile_layout,
                &ntt_module,
                "ntt_tile"
            )
        );
        let ntt_fused_layout =
            pipeline_layout(&device, &[&ntt_bind_group_layout, &ntt_fused_shift_bgl]);
        let ntt_fused_pipeline = timed!(
            "pipeline: NTT Fused",
            create_pipeline(
                &device,
                "NTT Fused",
                &ntt_fused_layout,
                &ntt_fused_module,
                "ntt_tile_with_shift"
            )
        );
        let ntt_global_stage_pipeline = timed!(
            "pipeline: NTT Global Stage",
            create_pipeline(
                &device,
                "NTT Global Stage",
                &ntt_global_layout,
                &ntt_module,
                "ntt_global_stage"
            )
        );
        let ntt_bitreverse_pipeline = timed!(
            "pipeline: NTT BitReverse",
            create_pipeline(
                &device,
                "NTT BitReverse",
                &ntt_global_layout,
                &ntt_module,
                "bitreverse_inplace"
            )
        );

        // Polynomial pipelines
        let coset_shift_layout = pipeline_layout(&device, &[&coset_shift_bind_group_layout]);
        let pointwise_layout = pipeline_layout(&device, &[&pointwise_poly_bind_group_layout]);
        let montgomery_layout = pipeline_layout(&device, &[&montgomery_bind_group_layout]);
        let coset_shift_pipeline = timed!(
            "pipeline: Coset Shift",
            create_pipeline(
                &device,
                "Coset Shift",
                &coset_shift_layout,
                &poly_ops_module,
                "coset_shift"
            )
        );
        let pointwise_poly_pipeline = timed!(
            "pipeline: Pointwise Poly",
            create_pipeline(
                &device,
                "Pointwise Poly",
                &pointwise_layout,
                &poly_ops_module,
                "pointwise_poly"
            )
        );
        let to_montgomery_pipeline = timed!(
            "pipeline: To Montgomery",
            create_pipeline(
                &device,
                "To Montgomery",
                &montgomery_layout,
                &poly_ops_module,
                "to_montgomery_array"
            )
        );
        let from_montgomery_pipeline = timed!(
            "pipeline: From Montgomery",
            create_pipeline(
                &device,
                "From Montgomery",
                &montgomery_layout,
                &poly_ops_module,
                "from_montgomery_array"
            )
        );

        // MSM pipelines
        let msm_agg_layout = pipeline_layout(&device, &[&msm_agg_bind_group_layout]);
        let msm_sum_layout = pipeline_layout(&device, &[&msm_sum_bind_group_layout]);
        let msm_weight_g1_layout = pipeline_layout(&device, &[&msm_weight_g1_bind_group_layout]);
        let msm_subsum_phase1_layout =
            pipeline_layout(&device, &[&msm_subsum_phase1_bind_group_layout]);
        let msm_subsum_phase2_layout =
            pipeline_layout(&device, &[&msm_subsum_phase2_bind_group_layout]);

        let msm_agg_g1_pipeline = timed!(
            "pipeline: MSM Agg G1",
            create_pipeline(
                &device,
                "MSM Agg G1",
                &msm_agg_layout,
                &msm_module,
                "aggregate_buckets_g1"
            )
        );
        let msm_sum_g1_pipeline = timed!(
            "pipeline: MSM Sum G1",
            create_pipeline(
                &device,
                "MSM Sum G1",
                &msm_sum_layout,
                &msm_module,
                "subsum_accumulation_g1"
            )
        );
        let msm_agg_g2_pipeline = timed!(
            "pipeline: MSM Agg G2",
            create_pipeline(
                &device,
                "MSM Agg G2",
                &msm_agg_layout,
                &msm_module,
                "aggregate_buckets_g2"
            )
        );
        let msm_sum_g2_pipeline = timed!(
            "pipeline: MSM Sum G2",
            create_pipeline(
                &device,
                "MSM Sum G2",
                &msm_sum_layout,
                &msm_module,
                "subsum_accumulation_g2"
            )
        );
        let msm_to_mont_g1_pipeline = timed!(
            "pipeline: MSM To Mont G1",
            create_pipeline(
                &device,
                "MSM To Montgomery G1",
                &montgomery_layout,
                &msm_module,
                "to_montgomery_bases_g1"
            )
        );
        let msm_to_mont_g2_pipeline = timed!(
            "pipeline: MSM To Mont G2",
            create_pipeline(
                &device,
                "MSM To Montgomery G2",
                &montgomery_layout,
                &msm_module,
                "to_montgomery_bases_g2"
            )
        );
        let msm_weight_g1_pipeline = timed!(
            "pipeline: MSM Weight G1",
            create_pipeline(
                &device,
                "MSM Weight G1",
                &msm_weight_g1_layout,
                &msm_module,
                "weight_buckets_g1"
            )
        );
        let msm_subsum_phase1_g1_pipeline = timed!(
            "pipeline: MSM Subsum Ph1 G1",
            create_pipeline(
                &device,
                "MSM Subsum Phase1 G1",
                &msm_subsum_phase1_layout,
                &msm_module,
                "subsum_phase1_g1"
            )
        );
        let msm_subsum_phase2_g1_pipeline = timed!(
            "pipeline: MSM Subsum Ph2 G1",
            create_pipeline(
                &device,
                "MSM Subsum Phase2 G1",
                &msm_subsum_phase2_layout,
                &msm_module,
                "subsum_phase2_g1"
            )
        );

        let msm_weight_g2_layout = pipeline_layout(&device, &[&msm_weight_g2_bind_group_layout]);
        let msm_weight_g2_pipeline = timed!(
            "pipeline: MSM Weight G2",
            create_pipeline(
                &device,
                "MSM Weight G2",
                &msm_weight_g2_layout,
                &msm_module,
                "weight_buckets_g2"
            )
        );
        let msm_subsum_phase1_g2_pipeline = timed!(
            "pipeline: MSM Subsum Ph1 G2",
            create_pipeline(
                &device,
                "MSM Subsum Phase1 G2",
                &msm_subsum_phase1_layout,
                &msm_module,
                "subsum_phase1_g2"
            )
        );
        let msm_subsum_phase2_g2_pipeline = timed!(
            "pipeline: MSM Subsum Ph2 G2",
            create_pipeline(
                &device,
                "MSM Subsum Phase2 G2",
                &msm_subsum_phase2_layout,
                &msm_module,
                "subsum_phase2_g2"
            )
        );

        let msm_reduce_layout = pipeline_layout(&device, &[&msm_reduce_bind_group_layout]);
        let msm_reduce_g1_pipeline = timed!(
            "pipeline: MSM Reduce G1",
            create_pipeline(
                &device,
                "MSM Reduce G1",
                &msm_reduce_layout,
                &msm_module,
                "reduce_sub_buckets_g1"
            )
        );
        let msm_reduce_g2_pipeline = timed!(
            "pipeline: MSM Reduce G2",
            create_pipeline(
                &device,
                "MSM Reduce G2",
                &msm_reduce_layout,
                &msm_module,
                "reduce_sub_buckets_g2"
            )
        );
        #[cfg(feature = "timing")]
        {
            let pipelines_total = pipelines_start.elapsed();
            eprintln!("\n[init] === GpuContext::new() summary ===");
            eprintln!(
                "[init] {:<30} {:>8.1}ms",
                "shader compilation",
                shader_total.as_secs_f64() * 1000.0
            );
            eprintln!(
                "[init] {:<30} {:>8.1}ms",
                "bind group layouts",
                layouts_total.as_secs_f64() * 1000.0
            );
            eprintln!(
                "[init] {:<30} {:>8.1}ms",
                "pipeline creation",
                pipelines_total.as_secs_f64() * 1000.0
            );
            eprintln!(
                "[init] {:<30} {:>8.1}ms",
                "TOTAL",
                init_start.elapsed().as_secs_f64() * 1000.0
            );
            eprintln!();
        }

        #[cfg(feature = "profiling")]
        let profiler = Mutex::new(wgpu_profiler::GpuProfiler::new(
            &device,
            wgpu_profiler::GpuProfilerSettings {
                enable_timer_queries: true,
                ..Default::default()
            },
        )?);

        Ok(Self {
            device,
            queue,
            ntt_pipeline,
            ntt_fused_pipeline,
            ntt_global_stage_pipeline,
            ntt_bitreverse_pipeline,
            coset_shift_pipeline,
            pointwise_poly_pipeline,
            to_montgomery_pipeline,
            from_montgomery_pipeline,
            msm_agg_g1_pipeline,
            msm_sum_g1_pipeline,
            msm_agg_g2_pipeline,
            msm_sum_g2_pipeline,
            msm_to_mont_g1_pipeline,
            msm_to_mont_g2_pipeline,
            msm_weight_g1_pipeline,
            msm_subsum_phase1_g1_pipeline,
            msm_subsum_phase2_g1_pipeline,
            msm_weight_g2_pipeline,
            msm_subsum_phase1_g2_pipeline,
            msm_subsum_phase2_g2_pipeline,
            msm_reduce_g1_pipeline,
            msm_reduce_g2_pipeline,
            ntt_bind_group_layout,
            ntt_fused_shift_bgl,
            ntt_params_bind_group_layout,
            coset_shift_bind_group_layout,
            pointwise_poly_bind_group_layout,
            montgomery_bind_group_layout,
            msm_agg_bind_group_layout,
            msm_sum_bind_group_layout,
            msm_weight_g1_bind_group_layout,
            msm_weight_g2_bind_group_layout,
            msm_subsum_phase1_bind_group_layout,
            msm_subsum_phase2_bind_group_layout,
            msm_reduce_bind_group_layout,
            _marker: PhantomData,
            #[cfg(feature = "profiling")]
            profiler,
        })
    }

    #[cfg(feature = "profiling")]
    pub fn end_profiler_frame(&self) {
        // Ensure all GPU work and timestamp query readbacks are complete
        // before ending the frame. Without this, Metal may return stale or
        // uninitialized timestamp data (causing negative durations).
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let mut profiler = self.profiler.lock().unwrap();
        profiler.end_frame().expect("end_frame failed");
    }

    #[cfg(feature = "profiling")]
    pub fn process_profiler_results(&self) -> Option<Vec<wgpu_profiler::GpuTimerQueryResult>> {
        let mut profiler = self.profiler.lock().unwrap();
        profiler.process_finished_frame(self.queue.get_timestamp_period())
    }

    pub fn create_storage_buffer(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }

    pub fn create_empty_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

#[cfg(test)]
mod tests;
