//! H polynomial GPU pipeline dispatcher.
//!
//! Computes the quotient polynomial H(x) = (A(x)·B(x) − C(x)) / Z(x) entirely
//! on the GPU using a single command encoder with the following steps:
//!
//! 1. **To Montgomery**: Convert A, B, C, twiddle factors, shift arrays, and Z⁻¹
//!    into Montgomery domain for efficient modular arithmetic
//! 2. **Fused iNTT + Coset shift(A, B, C)**: Inverse NTT with shift factors
//!    multiplied during write-back (avoids separate coset_shift dispatch)
//! 3. **NTT(A, B, C)**: Forward NTT to get evaluation representations on the coset
//! 4. **Pointwise H = (A·B − C) · Z⁻¹**: Element-wise computation in evaluation domain
//! 5. **Fused iNTT + Inverse coset shift(H)**: iNTT with inverse shift fused in
//! 6. **From Montgomery(H)**: Convert H out of Montgomery domain

use wgpu::util::DeviceExt;

use super::curve::GpuCurve;
use super::{GpuContext, HPolyBuffers, compute_pass};

impl<C: GpuCurve> GpuContext<C> {
    pub fn execute_h_pipeline(&self, bufs: &HPolyBuffers<'_>, n: u32) {
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

        let fused_shift_bg = |shifts: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("NTT Fused Shift BG"),
                layout: &self.ntt_fused_shift_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shifts.as_entire_binding(),
                }],
            })
        };

        let pointwise_fused_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pointwise Fused BG"),
            layout: &self.pointwise_fused_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bufs.a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bufs.b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bufs.c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bufs.z_invs.as_entire_binding(),
                },
            ],
        });

        #[cfg(feature = "profiling")]
        let mut profiler_guard = self.profiler.lock().unwrap();
        #[cfg(feature = "profiling")]
        let mut scope = profiler_guard.scope("h_pipeline", &mut encoder);

        // Keep uniform buffers alive until the queue is submitted!
        let mut param_updates = Vec::new();

        // 1. To Montgomery
        for bg in [
            mont_bg(bufs.a),
            mont_bg(bufs.b),
            mont_bg(bufs.c),
            mont_bg(bufs.twiddles_inv),
            mont_bg(bufs.twiddles_fwd),
            mont_bg(bufs.shifts),
            mont_bg(bufs.inv_shifts),
            mont_bg(bufs.z_invs),
        ] {
            let mut pass = compute_pass!(scope, encoder, "to_montgomery");
            pass.set_pipeline(&self.to_montgomery_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(C::SCALAR_WORKGROUP_SIZE), 1, 1);
        }

        macro_rules! encode_ntt {
            (
                $label:expr,
                $data_buf:expr,
                $tw_buf:expr,
                $is_fused_shift:expr,
                $shifts_buf:expr,
                $is_h_fused_pointwise:expr
            ) => {
                if n <= C::NTT_TILE_SIZE {
                    let bg = ntt_bg($data_buf, $tw_buf);
                    if $is_h_fused_pointwise {
                        let shifts_group1 = fused_shift_bg($shifts_buf.unwrap());
                        let mut pass = compute_pass!(scope, encoder, concat!($label, "_fused_h"));
                        pass.set_pipeline(&self.ntt_tile_fused_pointwise_pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.set_bind_group(1, &shifts_group1, &[]);
                        pass.set_bind_group(2, &pointwise_fused_bg, &[]);
                        pass.dispatch_workgroups(n.div_ceil(C::NTT_TILE_SIZE), 1, 1);
                    } else if $is_fused_shift {
                        let shifts_group1 = fused_shift_bg($shifts_buf.unwrap());
                        let mut pass = compute_pass!(scope, encoder, concat!($label, "_fused"));
                        pass.set_pipeline(&self.ntt_fused_pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.set_bind_group(1, &shifts_group1, &[]);
                        pass.dispatch_workgroups(n.div_ceil(C::NTT_TILE_SIZE), 1, 1);
                    } else {
                        let mut pass = compute_pass!(scope, encoder, $label);
                        pass.set_pipeline(&self.ntt_pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(n.div_ceil(C::NTT_TILE_SIZE), 1, 1);
                    }
                } else {
                    let mut log_n = 0u32;
                    let mut m = n;
                    while m > 1 {
                        log_n += 1;
                        m >>= 1;
                    }

                    let mut stage_params = [n, 0, log_n, 0];
                    let params_buf =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("NTT Params Buffer"),
                                contents: bytemuck::cast_slice(&stage_params),
                                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                            });

                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("NTT Global BG"),
                        layout: &self.ntt_params_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: $data_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: $tw_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buf.as_entire_binding(),
                            },
                        ],
                    });

                    if $is_h_fused_pointwise {
                        let shifts_group1 = fused_shift_bg($shifts_buf.unwrap());
                        let mut pass =
                            compute_pass!(scope, encoder, concat!($label, "_bitreverse_fused_h"));
                        pass.set_pipeline(&self.ntt_bitreverse_fused_pointwise_pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.set_bind_group(1, &shifts_group1, &[]);
                        pass.set_bind_group(2, &pointwise_fused_bg, &[]);
                        pass.dispatch_workgroups(n.div_ceil(C::SCALAR_WORKGROUP_SIZE), 1, 1);
                    } else {
                        let mut pass =
                            compute_pass!(scope, encoder, concat!($label, "_bitreverse"));
                        pass.set_pipeline(&self.ntt_bitreverse_pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(n.div_ceil(C::SCALAR_WORKGROUP_SIZE), 1, 1);
                    }

                    let mut half_len = 1u32;
                    if (log_n & 1) == 1 {
                        stage_params[1] = half_len;
                        let update_buf =
                            self.device
                                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("NTT Params Update"),
                                    contents: bytemuck::cast_slice(&stage_params),
                                    usage: wgpu::BufferUsages::COPY_SRC,
                                });
                        encoder.copy_buffer_to_buffer(&update_buf, 0, &params_buf, 0, 16);
                        param_updates.push(update_buf);

                        let mut pass = compute_pass!(scope, encoder, concat!($label, "_stage"));
                        pass.set_pipeline(&self.ntt_global_stage_pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups((n / 2).div_ceil(C::SCALAR_WORKGROUP_SIZE), 1, 1);

                        half_len = 2;
                    }

                    while half_len < n {
                        stage_params[1] = half_len;
                        let update_buf =
                            self.device
                                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("NTT Params Update"),
                                    contents: bytemuck::cast_slice(&stage_params),
                                    usage: wgpu::BufferUsages::COPY_SRC,
                                });
                        encoder.copy_buffer_to_buffer(&update_buf, 0, &params_buf, 0, 16);
                        param_updates.push(update_buf);

                        let mut pass =
                            compute_pass!(scope, encoder, concat!($label, "_stage_radix4"));
                        pass.set_pipeline(&self.ntt_global_stage_radix4_pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups((n / 4).div_ceil(C::SCALAR_WORKGROUP_SIZE), 1, 1);

                        half_len <<= 2;
                    }

                    if $is_fused_shift || $is_h_fused_pointwise {
                        let shift_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Coset Shift BG"),
                            layout: &self.coset_shift_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: $data_buf.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: $shifts_buf.unwrap().as_entire_binding(),
                                },
                            ],
                        });
                        let mut pass = compute_pass!(scope, encoder, concat!($label, "_shift"));
                        pass.set_pipeline(&self.coset_shift_pipeline);
                        pass.set_bind_group(0, &shift_bg, &[]);
                        pass.dispatch_workgroups(n.div_ceil(C::SCALAR_WORKGROUP_SIZE), 1, 1);
                    }

                    param_updates.push(params_buf);
                }
            };
        }

        // 2. Fused iNTT + coset shift on A/B/C
        encode_ntt!(
            "intt_a",
            bufs.a,
            bufs.twiddles_inv,
            true,
            Some(bufs.shifts),
            false
        );
        encode_ntt!(
            "intt_b",
            bufs.b,
            bufs.twiddles_inv,
            true,
            Some(bufs.shifts),
            false
        );
        encode_ntt!(
            "intt_c",
            bufs.c,
            bufs.twiddles_inv,
            true,
            Some(bufs.shifts),
            false
        );

        // 3. NTT on A/B/C
        encode_ntt!(
            "ntt_a",
            bufs.a,
            bufs.twiddles_fwd,
            false,
            None::<&wgpu::Buffer>,
            false
        );
        encode_ntt!(
            "ntt_b",
            bufs.b,
            bufs.twiddles_fwd,
            false,
            None::<&wgpu::Buffer>,
            false
        );
        encode_ntt!(
            "ntt_c",
            bufs.c,
            bufs.twiddles_fwd,
            false,
            None::<&wgpu::Buffer>,
            false
        );

        // 4/5. Fused pointwise + iNTT(H) + inverse coset shift
        encode_ntt!(
            "intt_h",
            bufs.h,
            bufs.twiddles_inv,
            false,
            Some(bufs.inv_shifts),
            true
        );

        // 6. From Montgomery on H
        {
            let bg = mont_bg(bufs.h);
            let mut pass = compute_pass!(scope, encoder, "from_montgomery_h");
            pass.set_pipeline(&self.from_montgomery_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(C::SCALAR_WORKGROUP_SIZE), 1, 1);
        }

        #[cfg(feature = "profiling")]
        {
            drop(scope);
            profiler_guard.resolve_queries(&mut encoder);
        }

        self.queue.submit(Some(encoder.finish()));

        // Ensure param buffers aren't dropped until the queue submission finishes
        drop(param_updates);
    }
}
