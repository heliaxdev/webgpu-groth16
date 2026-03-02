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

use anyhow::Result;
use ff::{Field, PrimeField};
use wgpu::util::DeviceExt;

use super::curve::GpuCurve;
use super::{GpuContext, HPolyBuffers, NTT_TILE_SIZE, SCALAR_WORKGROUP_SIZE, compute_pass};
use crate::prover::marshal_scalars;

pub(crate) struct HPolyPending {
    pub h_buf: wgpu::Buffer,
    pub n: usize,
}

pub(crate) fn submit_h_poly<G: GpuCurve>(
    gpu: &GpuContext<G>,
    a_values: &[G::Scalar],
    b_values: &[G::Scalar],
    c_values: &[G::Scalar],
) -> Result<HPolyPending> {
    let n = a_values.len().next_power_of_two();

    let omega_n = G::root_of_unity(n);
    let omega_n_inv = omega_n.invert().unwrap();

    let n_inv = G::Scalar::from(n as u64).invert().unwrap();
    let coset_generator = G::Scalar::MULTIPLICATIVE_GENERATOR;
    let coset_inv = coset_generator.invert().unwrap();

    let mut inv_twiddles_n = vec![G::Scalar::ONE; n];
    let mut fwd_twiddles_n = vec![G::Scalar::ONE; n];
    let mut shifts = vec![G::Scalar::ONE; n];
    let mut inv_shifts = vec![G::Scalar::ONE; n];

    for i in 1..n {
        inv_twiddles_n[i] = inv_twiddles_n[i - 1] * omega_n_inv;
        fwd_twiddles_n[i] = fwd_twiddles_n[i - 1] * omega_n;
        shifts[i] = shifts[i - 1] * coset_generator;
        inv_shifts[i] = inv_shifts[i - 1] * coset_inv;
    }

    for i in 0..n {
        shifts[i] *= n_inv;
        inv_shifts[i] *= n_inv;
    }

    let g_to_n = coset_generator.pow([n as u64]);
    let z_inv = (g_to_n - G::Scalar::ONE).invert().unwrap();
    let z_invs = vec![z_inv];

    let mut a_coeffs = a_values.to_vec();
    a_coeffs.resize(n, G::Scalar::ZERO);
    let mut b_coeffs = b_values.to_vec();
    b_coeffs.resize(n, G::Scalar::ZERO);
    let mut c_coeffs = c_values.to_vec();
    c_coeffs.resize(n, G::Scalar::ZERO);

    let a_buf = gpu.create_storage_buffer("A", &marshal_scalars::<G>(&a_coeffs));
    let b_buf = gpu.create_storage_buffer("B", &marshal_scalars::<G>(&b_coeffs));
    let c_buf = gpu.create_storage_buffer("C", &marshal_scalars::<G>(&c_coeffs));
    let h_buf = gpu.create_empty_buffer("H", (n * 32) as u64);

    let tw_inv_n_buf = gpu.create_storage_buffer("TwInvN", &marshal_scalars::<G>(&inv_twiddles_n));
    let tw_fwd_n_buf = gpu.create_storage_buffer("TwFwdN", &marshal_scalars::<G>(&fwd_twiddles_n));
    let shifts_buf = gpu.create_storage_buffer("Shifts", &marshal_scalars::<G>(&shifts));
    let inv_shifts_buf = gpu.create_storage_buffer("InvShifts", &marshal_scalars::<G>(&inv_shifts));
    let z_invs_buf = gpu.create_storage_buffer("ZInvs", &marshal_scalars::<G>(&z_invs));

    gpu.execute_h_pipeline(
        &HPolyBuffers {
            a: &a_buf,
            b: &b_buf,
            c: &c_buf,
            h: &h_buf,
            twiddles_inv: &tw_inv_n_buf,
            twiddles_fwd: &tw_fwd_n_buf,
            shifts: &shifts_buf,
            inv_shifts: &inv_shifts_buf,
            z_invs: &z_invs_buf,
        },
        n as u32,
    );

    Ok(HPolyPending { h_buf, n })
}

pub(crate) async fn read_h_poly_result<G: GpuCurve>(
    gpu: &GpuContext<G>,
    pending: HPolyPending,
) -> Result<Vec<G::Scalar>> {
    let h_bytes = gpu
        .read_buffer(&pending.h_buf, (pending.n * 32) as wgpu::BufferAddress)
        .await?;

    let mut h_poly = vec![G::Scalar::ZERO; pending.n];
    for (i, chunk) in h_bytes.chunks_exact(32).enumerate() {
        h_poly[i] = G::deserialize_scalar(chunk)?;
    }

    Ok(h_poly)
}

pub async fn compute_h_poly<G: GpuCurve>(
    gpu: &GpuContext<G>,
    a_values: &[G::Scalar],
    b_values: &[G::Scalar],
    c_values: &[G::Scalar],
) -> Result<Vec<G::Scalar>> {
    let pending = submit_h_poly::<G>(gpu, a_values, b_values, c_values)?;
    read_h_poly_result::<G>(gpu, pending).await
}

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

        let pointwise_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pointwise Poly BG"),
            layout: &self.pointwise_poly_bind_group_layout,
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
                    resource: bufs.h.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
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
            pass.dispatch_workgroups(n.div_ceil(SCALAR_WORKGROUP_SIZE), 1, 1);
        }

        // Helper macro to handle NTT scaling (Tile vs Global) dynamically
        macro_rules! encode_ntt {
            ($label:expr, $data_buf:expr, $tw_buf:expr, $is_fused:expr, $shifts_buf:expr) => {
                if n <= NTT_TILE_SIZE {
                    if $is_fused {
                        let bg = ntt_bg($data_buf, $tw_buf);
                        let shifts_group1 = fused_shift_bg($shifts_buf.unwrap());
                        let mut pass = compute_pass!(scope, encoder, concat!($label, "_fused"));
                        pass.set_pipeline(&self.ntt_fused_pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.set_bind_group(1, &shifts_group1, &[]);
                        pass.dispatch_workgroups(n.div_ceil(NTT_TILE_SIZE), 1, 1);
                    } else {
                        let bg = ntt_bg($data_buf, $tw_buf);
                        let mut pass = compute_pass!(scope, encoder, $label);
                        pass.set_pipeline(&self.ntt_pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(n.div_ceil(NTT_TILE_SIZE), 1, 1);
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

                    {
                        let mut pass =
                            compute_pass!(scope, encoder, concat!($label, "_bitreverse"));
                        pass.set_pipeline(&self.ntt_bitreverse_pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(n.div_ceil(SCALAR_WORKGROUP_SIZE), 1, 1);
                    }

                    let mut half_len = 1u32;
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

                        {
                            let mut pass = compute_pass!(scope, encoder, concat!($label, "_stage"));
                            pass.set_pipeline(&self.ntt_global_stage_pipeline);
                            pass.set_bind_group(0, &bg, &[]);
                            pass.dispatch_workgroups((n / 2).div_ceil(SCALAR_WORKGROUP_SIZE), 1, 1);
                        }
                        half_len <<= 1;
                    }

                    if $is_fused {
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
                        pass.dispatch_workgroups(n.div_ceil(SCALAR_WORKGROUP_SIZE), 1, 1);
                    }

                    param_updates.push(params_buf);
                }
            };
        }

        // 2. Fused iNTT + coset shift on A/B/C
        encode_ntt!("intt_a", bufs.a, bufs.twiddles_inv, true, Some(bufs.shifts));
        encode_ntt!("intt_b", bufs.b, bufs.twiddles_inv, true, Some(bufs.shifts));
        encode_ntt!("intt_c", bufs.c, bufs.twiddles_inv, true, Some(bufs.shifts));

        // 3. NTT on A/B/C
        encode_ntt!(
            "ntt_a",
            bufs.a,
            bufs.twiddles_fwd,
            false,
            None::<&wgpu::Buffer>
        );
        encode_ntt!(
            "ntt_b",
            bufs.b,
            bufs.twiddles_fwd,
            false,
            None::<&wgpu::Buffer>
        );
        encode_ntt!(
            "ntt_c",
            bufs.c,
            bufs.twiddles_fwd,
            false,
            None::<&wgpu::Buffer>
        );

        // 4. Pointwise H = (A*B-C)/Z
        {
            let mut pass = compute_pass!(scope, encoder, "pointwise_poly");
            pass.set_pipeline(&self.pointwise_poly_pipeline);
            pass.set_bind_group(0, &pointwise_bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(SCALAR_WORKGROUP_SIZE), 1, 1);
        }

        // 5. Fused iNTT + inverse coset shift on H
        encode_ntt!(
            "intt_h",
            bufs.h,
            bufs.twiddles_inv,
            true,
            Some(bufs.inv_shifts)
        );

        // 6. From Montgomery on H
        {
            let bg = mont_bg(bufs.h);
            let mut pass = compute_pass!(scope, encoder, "from_montgomery_h");
            pass.set_pipeline(&self.from_montgomery_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(SCALAR_WORKGROUP_SIZE), 1, 1);
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
