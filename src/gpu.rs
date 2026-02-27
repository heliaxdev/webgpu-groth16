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

    // Polynomial Pipelines
    pub ntt_pipeline: wgpu::ComputePipeline,
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

    // Bind Group Layouts
    pub ntt_bind_group_layout: wgpu::BindGroupLayout,
    pub ntt_params_bind_group_layout: wgpu::BindGroupLayout,
    pub coset_shift_bind_group_layout: wgpu::BindGroupLayout,
    pub pointwise_poly_bind_group_layout: wgpu::BindGroupLayout,
    pub montgomery_bind_group_layout: wgpu::BindGroupLayout,
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
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .context("Failed to find a compatible WebGPU adapter")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Groth16 Prover Device"),
                // Use adapter.limits() directly to support large buffers (>128MB) for WebAssembly
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .context("Failed to request WebGPU device")?;

        // 1. Compile Shader Modules
        let ntt_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("NTT Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(C::NTT_SOURCE)),
        });

        let msm_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MSM Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(C::MSM_SOURCE)),
        });

        let poly_ops_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Poly Ops Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(C::POLY_OPS_SOURCE)),
        });

        // 2. Define Bind Group Layouts
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

        let ntt_params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("NTT Global Bind Group Layout"),
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let coset_shift_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Coset Shift Bind Group Layout"),
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

        let pointwise_poly_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Pointwise Poly Bind Group Layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
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

        let montgomery_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Montgomery Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

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


        // 3. Create Compute Pipelines
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

        let ntt_global_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("NTT Global Pipeline Layout"),
            bind_group_layouts: &[&ntt_params_bind_group_layout],
            immediate_size: 0,
        });

        let ntt_global_stage_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("NTT Global Stage Pipeline"),
                layout: Some(&ntt_global_layout),
                module: &ntt_module,
                entry_point: Some("ntt_global_stage"),
                compilation_options: Default::default(),
                cache: None,
            });

        let ntt_bitreverse_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("NTT BitReverse Pipeline"),
                layout: Some(&ntt_global_layout),
                module: &ntt_module,
                entry_point: Some("bitreverse_inplace"),
                compilation_options: Default::default(),
                cache: None,
            });

        let coset_shift_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Coset Shift Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&coset_shift_bind_group_layout],
                        immediate_size: 0,
                    }),
                ),
                module: &poly_ops_module,
                entry_point: Some("coset_shift"),
                compilation_options: Default::default(),
                cache: None,
            });

        let pointwise_poly_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Pointwise Poly Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&pointwise_poly_bind_group_layout],
                        immediate_size: 0,
                    }),
                ),
                module: &poly_ops_module,
                entry_point: Some("pointwise_poly"),
                compilation_options: Default::default(),
                cache: None,
            });

        let to_montgomery_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("To Montgomery Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&montgomery_bind_group_layout],
                        immediate_size: 0,
                    }),
                ),
                module: &poly_ops_module,
                entry_point: Some("to_montgomery_array"),
                compilation_options: Default::default(),
                cache: None,
            });

        let from_montgomery_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("From Montgomery Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&montgomery_bind_group_layout],
                        immediate_size: 0,
                    }),
                ),
                module: &poly_ops_module,
                entry_point: Some("from_montgomery_array"),
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
            ntt_bind_group_layout,
            ntt_params_bind_group_layout,
            coset_shift_bind_group_layout,
            pointwise_poly_bind_group_layout,
            montgomery_bind_group_layout,
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

    pub fn execute_to_montgomery(&self, buffer: &wgpu::Buffer, num_elements: u32) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("To Montgomery Bind Group"),
            layout: &self.montgomery_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.to_montgomery_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(num_elements.div_ceil(256), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn execute_from_montgomery(&self, buffer: &wgpu::Buffer, num_elements: u32) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("From Montgomery Bind Group"),
            layout: &self.montgomery_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.from_montgomery_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(num_elements.div_ceil(256), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn execute_ntt(
        &self,
        data_buffer: &wgpu::Buffer,
        twiddles_buffer: &wgpu::Buffer,
        num_elements: u32,
    ) {
        if num_elements > 512 {
            self.execute_ntt_global(data_buffer, twiddles_buffer, num_elements);
            return;
        }

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

    pub fn execute_ntt_global(
        &self,
        data_buffer: &wgpu::Buffer,
        twiddles_buffer: &wgpu::Buffer,
        num_elements: u32,
    ) {
        let mut log_n = 0u32;
        let mut m = num_elements;
        while m > 1 {
            log_n += 1;
            m >>= 1;
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("NTT Global Encoder"),
            });

        let mut stage_params = [0u32; 4];
        stage_params[0] = num_elements;
        stage_params[2] = log_n;
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NTT Params Buffer"),
                contents: bytemuck::cast_slice(&stage_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let make_bind_group = |params_buf: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("NTT Global Bind Group"),
                layout: &self.ntt_params_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: twiddles_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            })
        };

        // Bit-reversal pass
        {
            let bg = make_bind_group(&params_buf);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("NTT BitReverse Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ntt_bitreverse_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(num_elements.div_ceil(256), 1, 1);
        }

        // Butterfly stages
        let mut half_len = 1u32;
        let mut param_updates: Vec<wgpu::Buffer> = Vec::new();
        while half_len < num_elements {
            stage_params[1] = half_len;
            let update_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("NTT Params Update"),
                    contents: bytemuck::cast_slice(&stage_params),
                    usage: wgpu::BufferUsages::COPY_SRC,
                });
            encoder.copy_buffer_to_buffer(
                &update_buf,
                0,
                &params_buf,
                0,
                16,
            );
            param_updates.push(update_buf);

            let bg = make_bind_group(&params_buf);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("NTT Global Stage Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ntt_global_stage_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((num_elements / 2).div_ceil(256), 1, 1);

            half_len <<= 1;
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn execute_coset_shift(
        &self,
        data_buffer: &wgpu::Buffer,
        shifts_buffer: &wgpu::Buffer,
        num_elements: u32,
    ) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Coset Shift Bind Group"),
            layout: &self.coset_shift_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: shifts_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.coset_shift_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(num_elements.div_ceil(256), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn execute_pointwise_poly(
        &self,
        a_buf: &wgpu::Buffer,
        b_buf: &wgpu::Buffer,
        c_buf: &wgpu::Buffer,
        h_buf: &wgpu::Buffer,
        z_invs_buf: &wgpu::Buffer,
        num_elements: u32,
    ) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pointwise Poly Bind Group"),
            layout: &self.pointwise_poly_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: h_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: z_invs_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pointwise_poly_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(num_elements.div_ceil(256), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_h_pipeline(
        &self,
        a_buf: &wgpu::Buffer,
        b_buf: &wgpu::Buffer,
        c_buf: &wgpu::Buffer,
        h_buf: &wgpu::Buffer,
        tw_inv_n_buf: &wgpu::Buffer,
        tw_fwd_n_buf: &wgpu::Buffer,
        shifts_buf: &wgpu::Buffer,
        inv_shifts_buf: &wgpu::Buffer,
        z_invs_buf: &wgpu::Buffer,
        n: u32,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("H Pipeline Encoder"),
            });

        let mont_bg = |buf: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Montgomery BG"),
                layout: &self.montgomery_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            })
        };

        let ntt_bg = |data: &wgpu::Buffer, tw: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("NTT BG"),
                layout: &self.ntt_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: tw.as_entire_binding(),
                    },
                ],
            })
        };

        let shift_bg = |data: &wgpu::Buffer, shifts: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Coset Shift BG"),
                layout: &self.coset_shift_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: shifts.as_entire_binding(),
                    },
                ],
            })
        };

        let pointwise_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pointwise Poly BG"),
            layout: &self.pointwise_poly_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: h_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: z_invs_buf.as_entire_binding(),
                },
            ],
        });

        // To Montgomery: a, b, c, twiddles (inv/fwd), shifts (fwd/inv), z_inv.
        for bg in [
            mont_bg(a_buf),
            mont_bg(b_buf),
            mont_bg(c_buf),
            mont_bg(tw_inv_n_buf),
            mont_bg(tw_fwd_n_buf),
            mont_bg(shifts_buf),
            mont_bg(inv_shifts_buf),
            mont_bg(z_invs_buf),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("To Montgomery Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.to_montgomery_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
        }

        // iNTT on A/B/C.
        for bg in [ntt_bg(a_buf, tw_inv_n_buf), ntt_bg(b_buf, tw_inv_n_buf), ntt_bg(c_buf, tw_inv_n_buf)] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("iNTT ABC Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ntt_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(512), 1, 1);
        }

        // Coset shift A/B/C.
        for bg in [
            shift_bg(a_buf, shifts_buf),
            shift_bg(b_buf, shifts_buf),
            shift_bg(c_buf, shifts_buf),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Coset Shift ABC Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.coset_shift_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
        }

        // NTT on A/B/C.
        for bg in [ntt_bg(a_buf, tw_fwd_n_buf), ntt_bg(b_buf, tw_fwd_n_buf), ntt_bg(c_buf, tw_fwd_n_buf)] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("NTT ABC Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ntt_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(512), 1, 1);
        }

        // Pointwise H = (A*B-C)/Z.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Pointwise Poly Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pointwise_poly_pipeline);
            pass.set_bind_group(0, &pointwise_bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
        }

        // iNTT(H).
        {
            let bg = ntt_bg(h_buf, tw_inv_n_buf);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("iNTT H Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ntt_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(512), 1, 1);
        }

        // Inverse coset shift on H.
        {
            let bg = shift_bg(h_buf, inv_shifts_buf);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Inv Coset Shift H Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.coset_shift_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
        }

        // From Montgomery on H.
        {
            let bg = mont_bg(h_buf);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("From Montgomery H Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.from_montgomery_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
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

        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        if let Ok(Ok(())) = receiver.await {
            let data = buffer_slice.get_mapped_range().to_vec();
            _ = buffer_slice;
            staging_buffer.unmap();
            return Ok(data);
        }
        anyhow::bail!("Failed to read back from GPU buffer")
    }

    pub async fn read_buffers_batch(
        &self,
        entries: &[(&wgpu::Buffer, wgpu::BufferAddress)],
    ) -> anyhow::Result<Vec<Vec<u8>>> {
        let mut staging = Vec::with_capacity(entries.len());
        for (_, size) in entries {
            staging.push(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Batch Staging Read Buffer"),
                size: *size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batch Read Encoder"),
            });
        for (i, (src, size)) in entries.iter().enumerate() {
            encoder.copy_buffer_to_buffer(src, 0, &staging[i], 0, *size);
        }
        self.queue.submit(Some(encoder.finish()));

        let mut receivers = Vec::with_capacity(staging.len());
        for s in &staging {
            let slice = s.slice(..);
            let (sender, receiver) = oneshot::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = sender.send(res);
            });
            receivers.push(receiver);
        }

        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        for r in receivers {
            match r.await {
                Ok(Ok(())) => {}
                _ => anyhow::bail!("Failed to map one of batch read buffers"),
            }
        }

        let mut out = Vec::with_capacity(staging.len());
        for s in staging {
            let bytes = s.slice(..).get_mapped_range().to_vec();
            s.unmap();
            out.push(bytes);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::curve::GpuCurve;
    use blstrs::{Bls12, G1Affine, Scalar};
    use ff::Field;
    use group::prime::PrimeCurveAffine;
    use std::borrow::Cow;

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

        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MSM Shader Roundtrip"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(<Bls12 as GpuCurve>::MSM_SOURCE)),
        });

        let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RT G1 BGL"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RT G1 Pipeline Layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RT G1 Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("roundtrip_g1"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RT G1 BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("RT G1 Encoder") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RT G1 Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        gpu.queue.submit(Some(encoder.finish()));

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
        let out_buf = gpu.create_empty_buffer("rt_out_coords_g1", in_bytes.len() as u64);

        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MSM Shader Coord Roundtrip"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(<Bls12 as GpuCurve>::MSM_SOURCE)),
        });

        let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RT Coords G1 BGL"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RT Coords G1 Pipeline Layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RT Coords G1 Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("roundtrip_coords_g1"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RT Coords G1 BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("RT Coords G1 Encoder") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RT Coords G1 Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        gpu.queue.submit(Some(encoder.finish()));

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
}
