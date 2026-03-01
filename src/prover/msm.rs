//! GPU-accelerated MSM (Multi-Scalar Multiplication) dispatch.
//!
//! Provides single-group MSM functions (`gpu_msm_g1`, `gpu_msm_g2`) and
//! batch dispatch (`gpu_msm_batch`, `gpu_msm_batch_bytes`) that enqueue
//! multiple MSMs into the GPU command queue before reading back results.

use anyhow::Result;

use crate::bucket::{BucketData, compute_bucket_sorting_with_width, compute_glv_bucket_sorting};
use crate::gpu::curve::{GpuCurve, G1_GPU_BYTES, G2_GPU_BYTES};
use crate::gpu::{GpuContext, MsmBuffers};

use super::prepared_key::{serialize_g1_bases, serialize_g1_phi_bases, serialize_g2_bases};

/// Holds GPU buffers uploaded from a BucketData, ready for MSM dispatch.
struct UploadedMsm {
    bases_buf: wgpu::Buffer,
    indices_buf: wgpu::Buffer,
    ptrs_buf: wgpu::Buffer,
    sizes_buf: wgpu::Buffer,
    vals_buf: wgpu::Buffer,
    w_starts_buf: wgpu::Buffer,
    w_counts_buf: wgpu::Buffer,
    agg_buf: wgpu::Buffer,
    sums_buf: wgpu::Buffer,
    reduce_starts_buf: Option<wgpu::Buffer>,
    reduce_counts_buf: Option<wgpu::Buffer>,
    orig_vals_buf: Option<wgpu::Buffer>,
    orig_wstarts_buf: Option<wgpu::Buffer>,
    orig_wcounts_buf: Option<wgpu::Buffer>,
}

impl UploadedMsm {
    fn as_msm_buffers(&self) -> MsmBuffers<'_> {
        MsmBuffers {
            bases: &self.bases_buf,
            base_indices: &self.indices_buf,
            bucket_pointers: &self.ptrs_buf,
            bucket_sizes: &self.sizes_buf,
            aggregated_buckets: &self.agg_buf,
            bucket_values: &self.vals_buf,
            window_starts: &self.w_starts_buf,
            window_counts: &self.w_counts_buf,
            window_sums: &self.sums_buf,
            reduce_starts: self.reduce_starts_buf.as_ref(),
            reduce_counts: self.reduce_counts_buf.as_ref(),
            orig_bucket_values: self.orig_vals_buf.as_ref(),
            orig_window_starts: self.orig_wstarts_buf.as_ref(),
            orig_window_counts: self.orig_wcounts_buf.as_ref(),
        }
    }
}

/// Uploads BucketData arrays to GPU buffers for MSM dispatch.
fn upload_bucket_data<G: GpuCurve>(
    gpu: &GpuContext<G>,
    name: &str,
    bases_bytes: &[u8],
    bd: &BucketData,
    point_gpu_bytes: usize,
) -> UploadedMsm {
    let bases_buf = gpu.create_storage_buffer(&format!("{name}_bases"), bases_bytes);
    let indices_buf = gpu.create_storage_buffer(
        &format!("{name}_indices"),
        bytemuck::cast_slice(&bd.base_indices),
    );
    let ptrs_buf = gpu.create_storage_buffer(
        &format!("{name}_ptrs"),
        bytemuck::cast_slice(&bd.bucket_pointers),
    );
    let sizes_buf = gpu.create_storage_buffer(
        &format!("{name}_sizes"),
        bytemuck::cast_slice(&bd.bucket_sizes),
    );
    let vals_buf = gpu.create_storage_buffer(
        &format!("{name}_vals"),
        bytemuck::cast_slice(&bd.bucket_values),
    );
    let w_starts_buf = gpu.create_storage_buffer(
        &format!("{name}_wstarts"),
        bytemuck::cast_slice(&bd.window_starts),
    );
    let w_counts_buf = gpu.create_storage_buffer(
        &format!("{name}_wcounts"),
        bytemuck::cast_slice(&bd.window_counts),
    );
    let agg_buf = gpu.create_empty_buffer(
        &format!("{name}_agg"),
        (bd.num_active_buckets as usize * point_gpu_bytes) as u64,
    );
    let sums_buf = gpu.create_empty_buffer(
        &format!("{name}_sums"),
        (bd.num_windows as usize * point_gpu_bytes) as u64,
    );

    let reduce_starts_buf = if bd.has_chunks {
        Some(gpu.create_storage_buffer(
            &format!("{name}_reduce_starts"),
            bytemuck::cast_slice(&bd.reduce_starts),
        ))
    } else {
        None
    };
    let reduce_counts_buf = if bd.has_chunks {
        Some(gpu.create_storage_buffer(
            &format!("{name}_reduce_counts"),
            bytemuck::cast_slice(&bd.reduce_counts),
        ))
    } else {
        None
    };
    let orig_vals_buf = if bd.has_chunks {
        Some(gpu.create_storage_buffer(
            &format!("{name}_orig_vals"),
            bytemuck::cast_slice(&bd.orig_bucket_values),
        ))
    } else {
        None
    };
    let orig_wstarts_buf = if bd.has_chunks {
        Some(gpu.create_storage_buffer(
            &format!("{name}_orig_wstarts"),
            bytemuck::cast_slice(&bd.orig_window_starts),
        ))
    } else {
        None
    };
    let orig_wcounts_buf = if bd.has_chunks {
        Some(gpu.create_storage_buffer(
            &format!("{name}_orig_wcounts"),
            bytemuck::cast_slice(&bd.orig_window_counts),
        ))
    } else {
        None
    };

    UploadedMsm {
        bases_buf,
        indices_buf,
        ptrs_buf,
        sizes_buf,
        vals_buf,
        w_starts_buf,
        w_counts_buf,
        agg_buf,
        sums_buf,
        reduce_starts_buf,
        reduce_counts_buf,
        orig_vals_buf,
        orig_wstarts_buf,
        orig_wcounts_buf,
    }
}

pub async fn gpu_msm_g1<G: GpuCurve>(
    gpu: &GpuContext<G>,
    bases: &[G::G1Affine],
    scalars: &[G::Scalar],
) -> Result<G::G1> {
    #[cfg(feature = "timing")]
    let t_start = std::time::Instant::now();

    let glv_c = G::glv_bucket_width();
    let bases_bytes = serialize_g1_bases::<G>(bases);
    let phi_bytes = serialize_g1_phi_bases::<G>(bases);
    let (glv_bytes, bd) = compute_glv_bucket_sorting::<G>(scalars, &bases_bytes, &phi_bytes, glv_c);
    if bd.num_active_buckets == 0 {
        return Ok(G::g1_identity());
    }

    #[cfg(feature = "timing")]
    let t_bucket = std::time::Instant::now();

    let uploaded = upload_bucket_data::<G>(gpu, "g1", &glv_bytes, &bd, G1_GPU_BYTES);

    #[cfg(feature = "timing")]
    let t_upload = std::time::Instant::now();

    gpu.execute_msm(
        false,
        &uploaded.as_msm_buffers(),
        bd.num_active_buckets,
        bd.num_dispatched,
        bd.has_chunks,
        bd.num_windows,
    );

    #[cfg(feature = "timing")]
    let t_dispatch = std::time::Instant::now();

    let result_bytes = gpu
        .read_buffer(&uploaded.sums_buf, (bd.num_windows as usize * G1_GPU_BYTES) as u64)
        .await?;

    #[cfg(feature = "timing")]
    let t_read = std::time::Instant::now();

    let result = fold_window_sums_g1::<G>(&result_bytes, bd.num_windows, bd.bucket_width)?;

    #[cfg(feature = "timing")]
    {
        let t_fold = std::time::Instant::now();
        eprintln!(
            "[timing] msm_g1 n={}: bucket_sort={:?} upload={:?} dispatch+gpu={:?} readback={:?} fold={:?} total={:?}",
            scalars.len(),
            t_bucket.duration_since(t_start),
            t_upload.duration_since(t_bucket),
            t_dispatch.duration_since(t_upload),
            t_read.duration_since(t_dispatch),
            t_fold.duration_since(t_read),
            t_fold.duration_since(t_start),
        );
    }

    Ok(result)
}

pub(crate) async fn gpu_msm_g2<G: GpuCurve>(
    gpu: &GpuContext<G>,
    bases: &[G::G2Affine],
    scalars: &[G::Scalar],
) -> Result<G::G2> {
    let bd: BucketData = compute_bucket_sorting_with_width::<G>(scalars, G::g2_bucket_width());
    if bd.num_active_buckets == 0 {
        return Ok(G::g2_identity());
    }

    let bases_bytes = serialize_g2_bases::<G>(bases);
    let uploaded = upload_bucket_data::<G>(gpu, "g2", &bases_bytes, &bd, G2_GPU_BYTES);

    gpu.execute_msm(
        true,
        &uploaded.as_msm_buffers(),
        bd.num_active_buckets,
        bd.num_dispatched,
        bd.has_chunks,
        bd.num_windows,
    );

    let result_bytes = gpu
        .read_buffer(&uploaded.sums_buf, (bd.num_windows as usize * G2_GPU_BYTES) as u64)
        .await?;
    fold_window_sums_g2::<G>(&result_bytes, bd.num_windows, bd.bucket_width)
}

/// Horner-style evaluation of window sums: processes windows from most-significant
/// to least-significant, doubling `bucket_width` times between each window.
fn fold_window_sums<P: Clone>(
    result_bytes: &[u8],
    num_windows: u32,
    bucket_width: usize,
    point_bytes: usize,
    identity: P,
    deserialize: impl Fn(&[u8]) -> Result<P>,
    add: impl Fn(&P, &P) -> P,
) -> Result<P> {
    let mut result = identity;
    for (i, chunk) in result_bytes.chunks_exact(point_bytes).enumerate().rev() {
        if i != (num_windows - 1) as usize {
            for _ in 0..bucket_width {
                result = add(&result, &result);
            }
        }
        let w_sum = deserialize(chunk)?;
        result = add(&result, &w_sum);
    }
    Ok(result)
}

pub(crate) fn fold_window_sums_g1<G: GpuCurve>(
    result_bytes: &[u8],
    num_windows: u32,
    bucket_width: usize,
) -> Result<G::G1> {
    fold_window_sums(
        result_bytes, num_windows, bucket_width, G1_GPU_BYTES,
        G::g1_identity(), G::deserialize_g1, G::add_g1_proj,
    )
}

fn fold_window_sums_g2<G: GpuCurve>(
    result_bytes: &[u8],
    num_windows: u32,
    bucket_width: usize,
) -> Result<G::G2> {
    fold_window_sums(
        result_bytes, num_windows, bucket_width, G2_GPU_BYTES,
        G::g2_identity(), G::deserialize_g2, G::add_g2_proj,
    )
}

#[allow(clippy::type_complexity)]
pub async fn gpu_msm_batch<G: GpuCurve>(
    gpu: &GpuContext<G>,
    a_bases: &[G::G1Affine],
    a_scalars: &[G::Scalar],
    b1_bases: &[G::G1Affine],
    b_scalars: &[G::Scalar],
    l_bases: &[G::G1Affine],
    l_scalars: &[G::Scalar],
    h_bases: &[G::G1Affine],
    h_scalars: &[G::Scalar],
    b2_bases: &[G::G2Affine],
    b2_scalars: &[G::Scalar],
) -> Result<(G::G1, G::G1, G::G1, G::G1, G::G2)> {
    let glv_c = G::glv_bucket_width();
    let a_bases_bytes = serialize_g1_bases::<G>(a_bases);
    let a_phi_bytes = serialize_g1_phi_bases::<G>(a_bases);
    let b1_bases_bytes = serialize_g1_bases::<G>(b1_bases);
    let b1_phi_bytes = serialize_g1_phi_bases::<G>(b1_bases);
    let l_bases_bytes = serialize_g1_bases::<G>(l_bases);
    let l_phi_bytes = serialize_g1_phi_bases::<G>(l_bases);
    let h_bases_bytes = serialize_g1_bases::<G>(h_bases);
    let h_phi_bytes = serialize_g1_phi_bases::<G>(h_bases);

    let (a_glv, a_bd) = compute_glv_bucket_sorting::<G>(a_scalars, &a_bases_bytes, &a_phi_bytes, glv_c);
    let (b1_glv, b_bd) = compute_glv_bucket_sorting::<G>(b_scalars, &b1_bases_bytes, &b1_phi_bytes, glv_c);
    let (l_glv, l_bd) = compute_glv_bucket_sorting::<G>(l_scalars, &l_bases_bytes, &l_phi_bytes, glv_c);
    let (h_glv, h_bd) = compute_glv_bucket_sorting::<G>(h_scalars, &h_bases_bytes, &h_phi_bytes, glv_c);
    let b2_bd = compute_bucket_sorting_with_width::<G>(b2_scalars, G::g2_bucket_width());
    gpu_msm_batch_bytes::<G>(
        gpu,
        &a_glv,
        a_bd,
        &b1_glv,
        b_bd,
        &l_glv,
        l_bd,
        &h_glv,
        h_bd,
        &serialize_g2_bases::<G>(b2_bases),
        b2_bd,
    )
    .await
}

/// Pending MSM job handle for batch dispatch.
struct MsmPending {
    sums_buf: wgpu::Buffer,
    num_windows: u32,
    bucket_width: usize,
}

#[allow(clippy::type_complexity)]
pub(crate) async fn gpu_msm_batch_bytes<G: GpuCurve>(
    gpu: &GpuContext<G>,
    a_bytes: &[u8],
    a_bd: BucketData,
    b1_bytes: &[u8],
    b1_bd: BucketData,
    l_bytes: &[u8],
    l_bd: BucketData,
    h_bytes: &[u8],
    h_bd: BucketData,
    b2_bytes: &[u8],
    b2_bd: BucketData,
) -> Result<(G::G1, G::G1, G::G1, G::G1, G::G2)> {
    let enqueue_g1 =
        |name: &str, bases_bytes: &[u8], bd: BucketData| -> Result<Option<MsmPending>> {
            if bd.num_active_buckets == 0 {
                return Ok(None);
            }
            let uploaded = upload_bucket_data::<G>(gpu, name, bases_bytes, &bd, G1_GPU_BYTES);
            gpu.execute_msm(
                false,
                &uploaded.as_msm_buffers(),
                bd.num_active_buckets,
                bd.num_dispatched,
                bd.has_chunks,
                bd.num_windows,
            );
            Ok(Some(MsmPending {
                sums_buf: uploaded.sums_buf,
                num_windows: bd.num_windows,
                bucket_width: bd.bucket_width,
            }))
        };

    let enqueue_g2 = |bases_bytes: &[u8], bd: BucketData| -> Result<Option<MsmPending>> {
        if bd.num_active_buckets == 0 {
            return Ok(None);
        }
        let uploaded = upload_bucket_data::<G>(gpu, "b2", bases_bytes, &bd, G2_GPU_BYTES);
        gpu.execute_msm(
            true,
            &uploaded.as_msm_buffers(),
            bd.num_active_buckets,
            bd.num_dispatched,
            bd.has_chunks,
            bd.num_windows,
        );
        Ok(Some(MsmPending {
            sums_buf: uploaded.sums_buf,
            num_windows: bd.num_windows,
            bucket_width: bd.bucket_width,
        }))
    };

    #[cfg(feature = "timing")]
    let t_msm_enqueue = std::time::Instant::now();
    let a_job = enqueue_g1("a", a_bytes, a_bd)?;
    #[cfg(feature = "timing")]
    eprintln!("[msm] enqueue a: {:?} ({} bases)", t_msm_enqueue.elapsed(), a_bytes.len() / G1_GPU_BYTES);
    #[cfg(feature = "timing")]
    let t_msm_enqueue = std::time::Instant::now();
    let b1_job = enqueue_g1("b1", b1_bytes, b1_bd)?;
    #[cfg(feature = "timing")]
    eprintln!("[msm] enqueue b1: {:?} ({} bases)", t_msm_enqueue.elapsed(), b1_bytes.len() / G1_GPU_BYTES);
    #[cfg(feature = "timing")]
    let t_msm_enqueue = std::time::Instant::now();
    let l_job = enqueue_g1("l", l_bytes, l_bd)?;
    #[cfg(feature = "timing")]
    eprintln!("[msm] enqueue l: {:?} ({} bases)", t_msm_enqueue.elapsed(), l_bytes.len() / G1_GPU_BYTES);
    #[cfg(feature = "timing")]
    let t_msm_enqueue = std::time::Instant::now();
    let h_job = enqueue_g1("h", h_bytes, h_bd)?;
    #[cfg(feature = "timing")]
    eprintln!("[msm] enqueue h: {:?} ({} bases)", t_msm_enqueue.elapsed(), h_bytes.len() / G1_GPU_BYTES);
    #[cfg(feature = "timing")]
    let t_msm_enqueue = std::time::Instant::now();
    let b2_job = enqueue_g2(b2_bytes, b2_bd)?;
    #[cfg(feature = "timing")]
    eprintln!("[msm] enqueue b2: {:?} ({} bases)", t_msm_enqueue.elapsed(), b2_bytes.len() / G2_GPU_BYTES);

    let mut read_targets: Vec<(&wgpu::Buffer, wgpu::BufferAddress)> = Vec::new();
    if let Some(job) = &a_job {
        read_targets.push((&job.sums_buf, (job.num_windows as usize * G1_GPU_BYTES) as u64));
    }
    if let Some(job) = &b1_job {
        read_targets.push((&job.sums_buf, (job.num_windows as usize * G1_GPU_BYTES) as u64));
    }
    if let Some(job) = &l_job {
        read_targets.push((&job.sums_buf, (job.num_windows as usize * G1_GPU_BYTES) as u64));
    }
    if let Some(job) = &h_job {
        read_targets.push((&job.sums_buf, (job.num_windows as usize * G1_GPU_BYTES) as u64));
    }
    if let Some(job) = &b2_job {
        read_targets.push((&job.sums_buf, (job.num_windows as usize * G2_GPU_BYTES) as u64));
    }

    #[cfg(feature = "timing")]
    let t_readback = std::time::Instant::now();
    let mut read_results = gpu.read_buffers_batch(&read_targets).await?.into_iter();
    #[cfg(feature = "timing")]
    eprintln!("[msm] readback: {:?}", t_readback.elapsed());

    let a = if let Some(job) = &a_job {
        fold_window_sums_g1::<G>(&read_results.next().unwrap(), job.num_windows, job.bucket_width)?
    } else {
        G::g1_identity()
    };
    let b1 = if let Some(job) = &b1_job {
        fold_window_sums_g1::<G>(&read_results.next().unwrap(), job.num_windows, job.bucket_width)?
    } else {
        G::g1_identity()
    };
    let l = if let Some(job) = &l_job {
        fold_window_sums_g1::<G>(&read_results.next().unwrap(), job.num_windows, job.bucket_width)?
    } else {
        G::g1_identity()
    };
    let h = if let Some(job) = &h_job {
        fold_window_sums_g1::<G>(&read_results.next().unwrap(), job.num_windows, job.bucket_width)?
    } else {
        G::g1_identity()
    };
    let b2 = if let Some(job) = &b2_job {
        fold_window_sums_g2::<G>(&read_results.next().unwrap(), job.num_windows, job.bucket_width)?
    } else {
        G::g2_identity()
    };

    Ok((a, b1, l, h, b2))
}
