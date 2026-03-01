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

use super::curve::GpuCurve;
use super::{compute_pass, GpuContext, HPolyBuffers, NTT_TILE_SIZE, SCALAR_WORKGROUP_SIZE};

impl<C: GpuCurve> GpuContext<C> {
    pub fn execute_h_pipeline(&self, bufs: &HPolyBuffers<'_>, n: u32) {
        let a_buf = bufs.a;
        let b_buf = bufs.b;
        let c_buf = bufs.c;
        let h_buf = bufs.h;
        let tw_inv_n_buf = bufs.twiddles_inv;
        let tw_fwd_n_buf = bufs.twiddles_fwd;
        let shifts_buf = bufs.shifts;
        let inv_shifts_buf = bufs.inv_shifts;
        let z_invs_buf = bufs.z_invs;
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

        #[cfg(feature = "profiling")]
        let mut profiler_guard = self.profiler.lock().unwrap();
        #[cfg(feature = "profiling")]
        let mut scope = profiler_guard.scope("h_pipeline", &mut encoder);

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
            let mut pass = compute_pass!(scope, encoder, "to_montgomery");
            pass.set_pipeline(&self.to_montgomery_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(SCALAR_WORKGROUP_SIZE), 1, 1);
        }

        // Fused iNTT + coset shift on A/B/C.
        let shifts_group1 = fused_shift_bg(shifts_buf);
        for bg in [
            ntt_bg(a_buf, tw_inv_n_buf),
            ntt_bg(b_buf, tw_inv_n_buf),
            ntt_bg(c_buf, tw_inv_n_buf),
        ] {
            let mut pass = compute_pass!(scope, encoder, "intt_shift_abc");
            pass.set_pipeline(&self.ntt_fused_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_bind_group(1, &shifts_group1, &[]);
            pass.dispatch_workgroups(n.div_ceil(NTT_TILE_SIZE), 1, 1);
        }

        // NTT on A/B/C.
        for bg in [
            ntt_bg(a_buf, tw_fwd_n_buf),
            ntt_bg(b_buf, tw_fwd_n_buf),
            ntt_bg(c_buf, tw_fwd_n_buf),
        ] {
            let mut pass = compute_pass!(scope, encoder, "ntt_abc");
            pass.set_pipeline(&self.ntt_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(NTT_TILE_SIZE), 1, 1);
        }

        // Pointwise H = (A*B-C)/Z.
        {
            let mut pass = compute_pass!(scope, encoder, "pointwise_poly");
            pass.set_pipeline(&self.pointwise_poly_pipeline);
            pass.set_bind_group(0, &pointwise_bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(SCALAR_WORKGROUP_SIZE), 1, 1);
        }

        // Fused iNTT + inverse coset shift on H.
        {
            let bg = ntt_bg(h_buf, tw_inv_n_buf);
            let inv_shifts_group1 = fused_shift_bg(inv_shifts_buf);
            let mut pass = compute_pass!(scope, encoder, "intt_shift_h");
            pass.set_pipeline(&self.ntt_fused_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_bind_group(1, &inv_shifts_group1, &[]);
            pass.dispatch_workgroups(n.div_ceil(NTT_TILE_SIZE), 1, 1);
        }

        // From Montgomery on H.
        {
            let bg = mont_bg(h_buf);
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
    }
}
