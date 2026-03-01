//! NTT and polynomial operation dispatchers.
//!
//! Contains GPU compute dispatch methods for:
//! - Montgomery domain conversion (to/from)
//! - Number Theoretic Transform (local tile and multi-stage global)
//! - Coset shift (multiply by powers of the multiplicative generator)
//! - Pointwise polynomial operations (H = (A·B − C) / Z)

use wgpu::util::DeviceExt;

use super::{GpuContext, SCALAR_WORKGROUP_SIZE, NTT_TILE_SIZE};
use super::curve::GpuCurve;

impl<C: GpuCurve> GpuContext<C> {
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
            cpass.dispatch_workgroups(num_elements.div_ceil(SCALAR_WORKGROUP_SIZE), 1, 1);
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
            cpass.dispatch_workgroups(num_elements.div_ceil(SCALAR_WORKGROUP_SIZE), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn execute_ntt(
        &self,
        data_buffer: &wgpu::Buffer,
        twiddles_buffer: &wgpu::Buffer,
        num_elements: u32,
    ) {
        if num_elements > NTT_TILE_SIZE {
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
            cpass.dispatch_workgroups(num_elements.div_ceil(NTT_TILE_SIZE), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Multi-stage global NTT for sizes > NTT_TILE_SIZE (512).
    ///
    /// Algorithm:
    /// 1. Bit-reversal permutation (in-place)
    /// 2. Iterative butterfly stages: for each `half_len` in 1, 2, 4, ..., n/2,
    ///    dispatches workgroups that combine pairs of elements using twiddle factors
    ///
    /// Each stage updates a uniform buffer with `[n, half_len, log_n, 0]` so the
    /// shader knows the current butterfly geometry.
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
            pass.dispatch_workgroups(num_elements.div_ceil(SCALAR_WORKGROUP_SIZE), 1, 1);
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
            encoder.copy_buffer_to_buffer(&update_buf, 0, &params_buf, 0, 16);
            param_updates.push(update_buf);

            let bg = make_bind_group(&params_buf);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("NTT Global Stage Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ntt_global_stage_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((num_elements / 2).div_ceil(SCALAR_WORKGROUP_SIZE), 1, 1);

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
            cpass.dispatch_workgroups(num_elements.div_ceil(SCALAR_WORKGROUP_SIZE), 1, 1);
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
            cpass.dispatch_workgroups(num_elements.div_ceil(SCALAR_WORKGROUP_SIZE), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}
