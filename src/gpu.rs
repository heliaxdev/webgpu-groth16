pub mod curve;

use std::borrow::Cow;
use std::marker::PhantomData;

use anyhow::Context;
use futures::channel::oneshot;
use wgpu::util::DeviceExt;

use self::curve::GpuCurve;

pub struct GpuContext<C> {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    pub ntt_pipeline: wgpu::ComputePipeline,

    // MSM 2-Stage Pipelines
    pub msm_agg_g1_pipeline: wgpu::ComputePipeline,
    pub msm_sum_g1_pipeline: wgpu::ComputePipeline,
    pub msm_agg_g2_pipeline: wgpu::ComputePipeline,
    pub msm_sum_g2_pipeline: wgpu::ComputePipeline,

    pub ntt_bind_group_layout: wgpu::BindGroupLayout,
    pub msm_agg_bind_group_layout: wgpu::BindGroupLayout,
    pub msm_sum_bind_group_layout: wgpu::BindGroupLayout,

    _marker: PhantomData<C>,
}

impl<C: GpuCurve> GpuContext<C> {
    pub async fn new() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .context("Failed to find a compatible WebGPU adapter")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Groth16 Prover Device"),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await?;

        let ntt_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("NTT Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(C::NTT_SOURCE)),
        });
        let msm_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MSM Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(C::MSM_SOURCE)),
        });

        let ntt_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("NTT Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Stage 1: Aggregation
        let msm_agg_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("MSM Agg Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Stage 2: Subsum Accumulation
        let msm_sum_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("MSM Sum Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let ntt_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NTT Compute Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&ntt_bind_group_layout],
                    immediate_size: 0,
                }),
            ),
            module: &ntt_module,
            entry_point: Some("ntt_tile"),
            compilation_options: Default::default(),
            cache: None,
        });

        let msm_agg_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&msm_agg_bind_group_layout],
            immediate_size: 0,
        });
        let msm_sum_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&msm_sum_bind_group_layout],
            immediate_size: 0,
        });

        let msm_agg_g1_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("MSM Agg G1"),
                layout: Some(&msm_agg_layout),
                module: &msm_module,
                entry_point: Some("aggregate_buckets_g1"),
                compilation_options: Default::default(),
                cache: None,
            });
        let msm_sum_g1_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("MSM Sum G1"),
                layout: Some(&msm_sum_layout),
                module: &msm_module,
                entry_point: Some("subsum_accumulation_g1"),
                compilation_options: Default::default(),
                cache: None,
            });
        let msm_agg_g2_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("MSM Agg G2"),
                layout: Some(&msm_agg_layout),
                module: &msm_module,
                entry_point: Some("aggregate_buckets_g2"),
                compilation_options: Default::default(),
                cache: None,
            });
        let msm_sum_g2_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("MSM Sum G2"),
                layout: Some(&msm_sum_layout),
                module: &msm_module,
                entry_point: Some("subsum_accumulation_g2"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            device,
            queue,
            ntt_pipeline,
            msm_agg_g1_pipeline,
            msm_sum_g1_pipeline,
            msm_agg_g2_pipeline,
            msm_sum_g2_pipeline,
            ntt_bind_group_layout,
            msm_agg_bind_group_layout,
            msm_sum_bind_group_layout,
            _marker: PhantomData,
        })
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

    pub fn execute_ntt(
        &self,
        data_buffer: &wgpu::Buffer,
        twiddles_buffer: &wgpu::Buffer,
        num_elements: u32,
    ) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NTT Bind Group"),
            layout: &self.ntt_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: twiddles_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("NTT Encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("NTT Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.ntt_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(num_elements.div_ceil(512), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_msm(
        &self,
        is_g2: bool,
        bases_buf: &wgpu::Buffer,
        base_indices_buf: &wgpu::Buffer,
        bucket_pointers_buf: &wgpu::Buffer,
        bucket_sizes_buf: &wgpu::Buffer,
        aggregated_buckets_buf: &wgpu::Buffer,
        bucket_values_buf: &wgpu::Buffer,
        window_starts_buf: &wgpu::Buffer,
        window_counts_buf: &wgpu::Buffer,
        window_sums_buf: &wgpu::Buffer,
        num_active_buckets: u32,
        num_windows: u32,
    ) {
        let agg_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MSM Agg Bind Group"),
            layout: &self.msm_agg_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bases_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: base_indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bucket_pointers_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bucket_sizes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: aggregated_buckets_buf.as_entire_binding(),
                },
            ],
        });

        let sum_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MSM Sum Bind Group"),
            layout: &self.msm_sum_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: aggregated_buckets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bucket_values_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: window_starts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: window_counts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: window_sums_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MSM Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MSM Pass 1"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(if is_g2 {
                &self.msm_agg_g2_pipeline
            } else {
                &self.msm_agg_g1_pipeline
            });
            cpass.set_bind_group(0, &agg_bind_group, &[]);
            cpass.dispatch_workgroups(num_active_buckets.div_ceil(64).max(1), 1, 1);
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MSM Pass 2"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(if is_g2 {
                &self.msm_sum_g2_pipeline
            } else {
                &self.msm_sum_g1_pipeline
            });
            cpass.set_bind_group(0, &sum_bind_group, &[]);
            cpass.dispatch_workgroups(num_windows, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub async fn read_buffer(
        &self,
        buffer: &wgpu::Buffer,
        size: wgpu::BufferAddress,
    ) -> anyhow::Result<Vec<u8>> {
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Read Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            sender.send(res).unwrap();
        });

        #[cfg(not(target_family = "wasm"))]
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        if let Ok(Ok(())) = receiver.await {
            let data = buffer_slice.get_mapped_range().to_vec();
            _ = buffer_slice;
            staging_buffer.unmap();
            return Ok(data);
        }
        anyhow::bail!("Failed to read back from GPU buffer")
    }
}
