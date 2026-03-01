//! MSM (Multi-Scalar Multiplication) GPU pipeline dispatcher.
//!
//! Executes the 5-kernel Pippenger MSM pipeline in a single command encoder:
//!
//! ```text
//! bases ──► [to_montgomery] ──► bases(mont)
//!                                    │
//! bucket_data ───────────────────────┤
//!                                    ▼
//!                            [aggregate_buckets] ──► agg_output
//!                                                        │
//!                               (if has_chunks)          ▼
//!                                              [reduce_sub_buckets] ──► aggregated_buckets
//!                                                                            │
//! bucket_values ─────────────────────────────────────────────────────────────┤
//!                                                                            ▼
//!                                                                    [weight_buckets]
//!                                                                            │
//!                                                                            ▼
//!                                                                    [subsum_phase1] ──► partial_sums
//!                                                                                             │
//!                                                                                             ▼
//!                                                                                     [subsum_phase2] ──► window_sums
//! ```
//!
//! When sub-bucket chunking is active (`has_chunks`), aggregate writes to an
//! intermediate buffer and a reduce pass sums sub-bucket partials into the
//! final per-bucket buffer before weighting.

use wgpu::util::DeviceExt;

use super::curve::{G1_GPU_BYTES, G2_GPU_BYTES, GpuCurve};
use super::{
    G1_SUBSUM_CHUNKS_PER_WINDOW, G2_SUBSUM_CHUNKS_PER_WINDOW, GpuContext, MSM_WORKGROUP_SIZE,
    MsmBuffers, compute_pass,
};

impl<C: GpuCurve> GpuContext<C> {
    #[allow(clippy::too_many_arguments)]
    pub fn execute_msm(
        &self,
        is_g2: bool,
        bufs: &MsmBuffers<'_>,
        num_active_buckets: u32,
        num_dispatched: u32,
        has_chunks: bool,
        num_windows: u32,
        skip_montgomery: bool,
    ) {
        let bases_buf = bufs.bases;
        let base_indices_buf = bufs.base_indices;
        let bucket_pointers_buf = bufs.bucket_pointers;
        let bucket_sizes_buf = bufs.bucket_sizes;
        let aggregated_buckets_buf = bufs.aggregated_buckets;
        let bucket_values_buf = bufs.bucket_values;
        let window_starts_buf = bufs.window_starts;
        let window_counts_buf = bufs.window_counts;
        let window_sums_buf = bufs.window_sums;

        let point_gpu_bytes: u64 = if is_g2 {
            G2_GPU_BYTES as u64
        } else {
            G1_GPU_BYTES as u64
        };

        // When chunking is active, aggregate writes to a larger intermediate buffer
        // and a reduce pass sums sub-buckets into the final aggregated_buckets buffer.
        let intermediate_buf = if has_chunks {
            Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MSM Intermediate Sub-Buckets"),
                size: num_dispatched as u64 * point_gpu_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        // The buffer the aggregate kernel writes to: intermediate (chunked) or final (unchunked).
        let agg_output_buf = intermediate_buf.as_ref().unwrap_or(aggregated_buckets_buf);

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
                    resource: agg_output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bucket_values_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MSM Encoder"),
            });

        #[cfg(feature = "profiling")]
        let mut profiler_guard = self.profiler.lock().unwrap();
        #[cfg(feature = "profiling")]
        let mut scope = profiler_guard.scope(if is_g2 { "msm_g2" } else { "msm_g1" }, &mut encoder);

        // Pre-pass: convert bases to Montgomery form in-place so aggregate
        // can skip per-point to_montgomery calls (saves 3 muls/load for G1, 6 for G2).
        // Skipped when using persistent bases that are already in Montgomery form.
        if !skip_montgomery {
            let mont_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MSM Bases Mont Bind Group"),
                layout: &self.montgomery_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bases_buf.as_entire_binding(),
                }],
            });
            let point_size: u64 = if is_g2 {
                G2_GPU_BYTES as u64
            } else {
                G1_GPU_BYTES as u64
            };
            let num_bases = (bases_buf.size() / point_size) as u32;
            let mut cpass = compute_pass!(scope, encoder, "to_montgomery_bases");
            cpass.set_pipeline(if is_g2 {
                &self.msm_to_mont_g2_pipeline
            } else {
                &self.msm_to_mont_g1_pipeline
            });
            cpass.set_bind_group(0, &mont_bind_group, &[]);
            cpass.dispatch_workgroups(num_bases.div_ceil(MSM_WORKGROUP_SIZE), 1, 1);
        }

        {
            let mut cpass = compute_pass!(scope, encoder, "bucket_aggregation");
            cpass.set_pipeline(if is_g2 {
                &self.msm_agg_g2_pipeline
            } else {
                &self.msm_agg_g1_pipeline
            });
            cpass.set_bind_group(0, &agg_bind_group, &[]);
            cpass.dispatch_workgroups(num_dispatched.div_ceil(MSM_WORKGROUP_SIZE).max(1), 1, 1);
        }

        // When sub-bucket chunking is active, reduce sub-bucket partial sums
        // into the final per-bucket aggregated results.
        if has_chunks {
            let reduce_starts_buf = bufs
                .reduce_starts
                .expect("reduce_starts required when has_chunks");
            let reduce_counts_buf = bufs
                .reduce_counts
                .expect("reduce_counts required when has_chunks");
            let reduce_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MSM Reduce Sub-Buckets BG"),
                layout: &self.msm_reduce_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: agg_output_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: reduce_starts_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: reduce_counts_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: aggregated_buckets_buf.as_entire_binding(),
                    },
                ],
            });
            let mut cpass = compute_pass!(scope, encoder, "reduce_sub_buckets");
            cpass.set_pipeline(if is_g2 {
                &self.msm_reduce_g2_pipeline
            } else {
                &self.msm_reduce_g1_pipeline
            });
            cpass.set_bind_group(0, &reduce_bind_group, &[]);
            cpass.dispatch_workgroups(num_active_buckets.div_ceil(MSM_WORKGROUP_SIZE).max(1), 1, 1);
        }

        // Weight each bucket sum by its bucket value in a separate kernel.
        // When chunking is active, use original bucket values (not sub-bucket values).
        let weight_values_buf = if has_chunks {
            bufs.orig_bucket_values
                .expect("orig_bucket_values required when has_chunks")
        } else {
            bucket_values_buf
        };
        {
            let weight_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(if is_g2 {
                    "MSM Weight G2 BG"
                } else {
                    "MSM Weight G1 BG"
                }),
                layout: if is_g2 {
                    &self.msm_weight_g2_bind_group_layout
                } else {
                    &self.msm_weight_g1_bind_group_layout
                },
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: aggregated_buckets_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: weight_values_buf.as_entire_binding(),
                    },
                ],
            });
            let mut cpass = compute_pass!(scope, encoder, "bucket_weighting");
            cpass.set_pipeline(if is_g2 {
                &self.msm_weight_g2_pipeline
            } else {
                &self.msm_weight_g1_pipeline
            });
            cpass.set_bind_group(0, &weight_bind_group, &[]);
            cpass.dispatch_workgroups(num_active_buckets.div_ceil(MSM_WORKGROUP_SIZE).max(1), 1, 1);
        }

        // Both G1 and G2: two-pass multi-workgroup tree reduction.
        // Phase 1: chunks_per_window workgroups per window each sum a contiguous
        //          slice of weighted buckets → partial_sums buffer.
        // Phase 2: one workgroup per window reduces partial_sums → final window_sums.
        //
        // When chunking is active, subsum must use original window metadata
        // (which maps to num_active_buckets layout in aggregated_buckets_buf).
        {
            let chunks_per_window = if is_g2 {
                G2_SUBSUM_CHUNKS_PER_WINDOW
            } else {
                G1_SUBSUM_CHUNKS_PER_WINDOW
            };
            let subsum_window_starts = if has_chunks {
                bufs.orig_window_starts
                    .expect("orig_window_starts required when has_chunks")
            } else {
                window_starts_buf
            };
            let subsum_window_counts = if has_chunks {
                bufs.orig_window_counts
                    .expect("orig_window_counts required when has_chunks")
            } else {
                window_counts_buf
            };

            let partial_sums_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MSM Partial Sums"),
                size: (num_windows * chunks_per_window) as u64 * point_gpu_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let subsum_params: [u32; 4] = [chunks_per_window, 0, 0, 0];
            let subsum_params_buf =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Subsum Params"),
                        contents: bytemuck::cast_slice(&subsum_params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let phase1_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MSM Subsum Phase1 BG"),
                layout: &self.msm_subsum_phase1_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: aggregated_buckets_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: subsum_window_starts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: subsum_window_counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: partial_sums_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: subsum_params_buf.as_entire_binding(),
                    },
                ],
            });

            let phase2_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MSM Subsum Phase2 BG"),
                layout: &self.msm_subsum_phase2_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: partial_sums_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: window_sums_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: subsum_params_buf.as_entire_binding(),
                    },
                ],
            });

            // Phase 1: many workgroups per window → partial sums.
            {
                let mut cpass = compute_pass!(scope, encoder, "tree_reduction_ph1");
                cpass.set_pipeline(if is_g2 {
                    &self.msm_subsum_phase1_g2_pipeline
                } else {
                    &self.msm_subsum_phase1_g1_pipeline
                });
                cpass.set_bind_group(0, &phase1_bind_group, &[]);
                cpass.dispatch_workgroups(num_windows * chunks_per_window, 1, 1);
            }

            // Phase 2: reduce partial sums → final window sums.
            {
                let mut cpass = compute_pass!(scope, encoder, "tree_reduction_ph2");
                cpass.set_pipeline(if is_g2 {
                    &self.msm_subsum_phase2_g2_pipeline
                } else {
                    &self.msm_subsum_phase2_g1_pipeline
                });
                cpass.set_bind_group(0, &phase2_bind_group, &[]);
                cpass.dispatch_workgroups(num_windows, 1, 1);
            }
        }

        #[cfg(feature = "profiling")]
        {
            drop(scope);
            profiler_guard.resolve_queries(&mut encoder);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Convert a bases buffer to Montgomery form in-place (one-time, for persistent bases).
    pub fn convert_to_montgomery(&self, buf: &wgpu::Buffer, is_g2: bool) {
        let mont_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Convert To Montgomery BG"),
            layout: &self.montgomery_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });
        let point_size: u64 = if is_g2 {
            G2_GPU_BYTES as u64
        } else {
            G1_GPU_BYTES as u64
        };
        let num_bases = (buf.size() / point_size) as u32;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Convert To Montgomery Encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("to_montgomery"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(if is_g2 {
                &self.msm_to_mont_g2_pipeline
            } else {
                &self.msm_to_mont_g1_pipeline
            });
            cpass.set_bind_group(0, &mont_bind_group, &[]);
            cpass.dispatch_workgroups(num_bases.div_ceil(MSM_WORKGROUP_SIZE), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}
