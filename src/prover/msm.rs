//! GPU-accelerated MSM (Multi-Scalar Multiplication) dispatch.
//!
//! Provides single-group MSM functions (`gpu_msm_g1`, `gpu_msm_g2`) and
//! batch dispatch (`gpu_msm_batch`) that enqueue multiple MSMs into the
//! GPU command queue before reading back results.

use anyhow::Result;

use super::prepared_key::{
    serialize_g1_bases, serialize_g1_phi_bases, serialize_g2_bases,
};
use crate::bucket::{
    BucketData, compute_bucket_sorting_with_width, compute_glv_bucket_sorting,
    optimal_glv_c,
};
use crate::gpu::curve::GpuCurve;
use crate::gpu::{GpuContext, MsmBuffers};

/// Source of point bases for an MSM dispatch.
pub(crate) enum MsmBases<'a> {
    /// Raw bytes to upload to GPU (needs Montgomery conversion).
    Bytes(&'a [u8]),
    /// Pre-uploaded persistent GPU buffer (already in Montgomery form).
    Persistent(&'a wgpu::Buffer),
}

/// Holds GPU buffers uploaded from a BucketData, ready for MSM dispatch.
struct UploadedMsm {
    /// Owned bases buffer (only set when bases are uploaded fresh).
    bases_buf: Option<wgpu::Buffer>,
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
    fn as_msm_buffers<'a>(
        &'a self,
        bases_override: Option<&'a wgpu::Buffer>,
    ) -> MsmBuffers<'a> {
        let bases = bases_override
            .or(self.bases_buf.as_ref())
            .expect("bases must be provided either as owned or persistent");
        MsmBuffers {
            bases,
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

/// Uploads BucketData arrays (and optionally bases) to GPU buffers for MSM
/// dispatch.
fn upload_msm_data<G: GpuCurve>(
    gpu: &GpuContext<G>,
    name: &str,
    bases_bytes: Option<&[u8]>,
    bd: &BucketData,
    point_gpu_bytes: usize,
) -> UploadedMsm {
    let bases_buf = bases_bytes
        .map(|b| gpu.create_storage_buffer(&format!("{name}_bases"), b));
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

    let glv_c = optimal_glv_c::<G>(scalars.len());
    let bases_bytes = serialize_g1_bases::<G>(bases);
    let phi_bytes = if G::HAS_G1_GLV {
        serialize_g1_phi_bases::<G>(bases)
    } else {
        Vec::new()
    };
    let (glv_bytes, bd) = compute_glv_bucket_sorting::<G>(
        scalars,
        &bases_bytes,
        &phi_bytes,
        glv_c,
    );

    #[cfg(feature = "timing")]
    let t_bucket = std::time::Instant::now();

    let pending =
        enqueue_msm::<G>(gpu, "g1", MsmBases::Bytes(&glv_bytes), bd, false)?;
    let job = match pending {
        None => return Ok(G::g1_identity()),
        Some(j) => j,
    };

    #[cfg(feature = "timing")]
    let t_enqueue = std::time::Instant::now();

    let result_bytes = gpu
        .read_buffer(
            &job.sums_buf,
            (job.num_windows as usize * G::G1_GPU_BYTES) as u64,
        )
        .await?;

    #[cfg(feature = "timing")]
    let t_read = std::time::Instant::now();

    let result = fold_window_sums_g1::<G>(
        &result_bytes,
        job.num_windows,
        job.bucket_width,
    )?;

    #[cfg(feature = "timing")]
    {
        let t_fold = std::time::Instant::now();
        eprintln!(
            "[timing] msm_g1 n={}: bucket_sort={:?} enqueue={:?} \
             readback={:?} fold={:?} total={:?}",
            scalars.len(),
            t_bucket.duration_since(t_start),
            t_enqueue.duration_since(t_bucket),
            t_read.duration_since(t_enqueue),
            t_fold.duration_since(t_read),
            t_fold.duration_since(t_start),
        );
    }

    Ok(result)
}

#[cfg(test)]
pub(crate) async fn gpu_msm_g2<G: GpuCurve>(
    gpu: &GpuContext<G>,
    bases: &[G::G2Affine],
    scalars: &[G::Scalar],
) -> Result<G::G2> {
    let bd =
        compute_bucket_sorting_with_width::<G>(scalars, G::g2_bucket_width());
    let bases_bytes = serialize_g2_bases::<G>(bases);

    let pending =
        enqueue_msm::<G>(gpu, "g2", MsmBases::Bytes(&bases_bytes), bd, true)?;
    let job = match pending {
        None => return Ok(G::g2_identity()),
        Some(j) => j,
    };

    let result_bytes = gpu
        .read_buffer(
            &job.sums_buf,
            (job.num_windows as usize * G::G2_GPU_BYTES) as u64,
        )
        .await?;
    fold_window_sums_g2::<G>(&result_bytes, job.num_windows, job.bucket_width)
}

/// Horner-style evaluation of window sums: processes windows from
/// most-significant to least-significant, doubling `bucket_width` times between
/// each window.
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
        result_bytes,
        num_windows,
        bucket_width,
        G::G1_GPU_BYTES,
        G::g1_identity(),
        G::deserialize_g1,
        G::add_g1_proj,
    )
}

fn fold_window_sums_g2<G: GpuCurve>(
    result_bytes: &[u8],
    num_windows: u32,
    bucket_width: usize,
) -> Result<G::G2> {
    fold_window_sums(
        result_bytes,
        num_windows,
        bucket_width,
        G::G2_GPU_BYTES,
        G::g2_identity(),
        G::deserialize_g2,
        G::add_g2_proj,
    )
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
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
    let a_bases_bytes = serialize_g1_bases::<G>(a_bases);
    let a_phi_bytes = if G::HAS_G1_GLV {
        serialize_g1_phi_bases::<G>(a_bases)
    } else {
        Vec::new()
    };
    let b1_bases_bytes = serialize_g1_bases::<G>(b1_bases);
    let b1_phi_bytes = if G::HAS_G1_GLV {
        serialize_g1_phi_bases::<G>(b1_bases)
    } else {
        Vec::new()
    };
    let l_bases_bytes = serialize_g1_bases::<G>(l_bases);
    let l_phi_bytes = if G::HAS_G1_GLV {
        serialize_g1_phi_bases::<G>(l_bases)
    } else {
        Vec::new()
    };
    let h_bases_bytes = serialize_g1_bases::<G>(h_bases);
    let h_phi_bytes = if G::HAS_G1_GLV {
        serialize_g1_phi_bases::<G>(h_bases)
    } else {
        Vec::new()
    };

    let a_c = optimal_glv_c::<G>(a_scalars.len());
    let b1_c = optimal_glv_c::<G>(b_scalars.len());
    let l_c = optimal_glv_c::<G>(l_scalars.len());
    let h_c = optimal_glv_c::<G>(h_scalars.len());
    let (a_glv, a_bd) = compute_glv_bucket_sorting::<G>(
        a_scalars,
        &a_bases_bytes,
        &a_phi_bytes,
        a_c,
    );
    let (b1_glv, b_bd) = compute_glv_bucket_sorting::<G>(
        b_scalars,
        &b1_bases_bytes,
        &b1_phi_bytes,
        b1_c,
    );
    let (l_glv, l_bd) = compute_glv_bucket_sorting::<G>(
        l_scalars,
        &l_bases_bytes,
        &l_phi_bytes,
        l_c,
    );
    let (h_glv, h_bd) = compute_glv_bucket_sorting::<G>(
        h_scalars,
        &h_bases_bytes,
        &h_phi_bytes,
        h_c,
    );
    let b2_bd = compute_bucket_sorting_with_width::<G>(
        b2_scalars,
        G::g2_bucket_width(),
    );

    let a_job =
        enqueue_msm::<G>(gpu, "a", MsmBases::Bytes(&a_glv), a_bd, false)?;
    let b1_job =
        enqueue_msm::<G>(gpu, "b1", MsmBases::Bytes(&b1_glv), b_bd, false)?;
    let l_job =
        enqueue_msm::<G>(gpu, "l", MsmBases::Bytes(&l_glv), l_bd, false)?;
    let h_job =
        enqueue_msm::<G>(gpu, "h", MsmBases::Bytes(&h_glv), h_bd, false)?;
    let b2_job = enqueue_msm::<G>(
        gpu,
        "b2",
        MsmBases::Bytes(&serialize_g2_bases::<G>(b2_bases)),
        b2_bd,
        true,
    )?;

    readback_msms::<G>(gpu, a_job, b1_job, l_job, h_job, b2_job).await
}

/// Pending MSM job handle for batch dispatch.
pub(crate) struct MsmPending {
    sums_buf: wgpu::Buffer,
    num_windows: u32,
    bucket_width: usize,
}

/// Enqueue a single MSM onto the GPU command queue (non-blocking).
///
/// Handles both uploaded (fresh) and persistent (pre-uploaded) bases via
/// `MsmBases`. Returns `None` when the bucket data has no active buckets
/// (all-zero scalars).
pub(crate) fn enqueue_msm<G: GpuCurve>(
    gpu: &GpuContext<G>,
    name: &str,
    bases: MsmBases<'_>,
    bd: BucketData,
    is_g2: bool,
) -> Result<Option<MsmPending>> {
    if bd.num_active_buckets == 0 {
        return Ok(None);
    }
    let point_size = if is_g2 {
        G::G2_GPU_BYTES
    } else {
        G::G1_GPU_BYTES
    };
    let (bases_bytes, bases_override, skip_montgomery) = match &bases {
        MsmBases::Bytes(b) => (Some(*b), None, false),
        MsmBases::Persistent(buf) => (None, Some(*buf), true),
    };
    let uploaded =
        upload_msm_data::<G>(gpu, name, bases_bytes, &bd, point_size);
    gpu.execute_msm(
        is_g2,
        &uploaded.as_msm_buffers(bases_override),
        bd.num_active_buckets,
        bd.num_dispatched,
        bd.has_chunks,
        bd.num_windows,
        skip_montgomery,
    );
    Ok(Some(MsmPending {
        sums_buf: uploaded.sums_buf,
        num_windows: bd.num_windows,
        bucket_width: bd.bucket_width,
    }))
}

/// Read back and fold results for 5 MSM jobs (4 G1 + 1 G2).
#[allow(clippy::type_complexity, clippy::manual_flatten)]
pub(crate) async fn readback_msms<G: GpuCurve>(
    gpu: &GpuContext<G>,
    a_job: Option<MsmPending>,
    b1_job: Option<MsmPending>,
    l_job: Option<MsmPending>,
    h_job: Option<MsmPending>,
    b2_job: Option<MsmPending>,
) -> Result<(G::G1, G::G1, G::G1, G::G1, G::G2)> {
    let g1_jobs = [&a_job, &b1_job, &l_job, &h_job];

    let mut read_targets: Vec<(&wgpu::Buffer, wgpu::BufferAddress)> =
        Vec::new();
    for job in &g1_jobs {
        if let Some(j) = job {
            read_targets.push((
                &j.sums_buf,
                (j.num_windows as usize * G::G1_GPU_BYTES) as u64,
            ));
        }
    }
    if let Some(j) = &b2_job {
        read_targets.push((
            &j.sums_buf,
            (j.num_windows as usize * G::G2_GPU_BYTES) as u64,
        ));
    }

    #[cfg(feature = "timing")]
    let t_readback = std::time::Instant::now();
    let mut read_results =
        gpu.read_buffers_batch(&read_targets).await?.into_iter();
    #[cfg(feature = "timing")]
    eprintln!("[msm] readback: {:?}", t_readback.elapsed());

    let mut fold_g1 = |job: &Option<MsmPending>| -> Result<G::G1> {
        match job {
            Some(j) => fold_window_sums_g1::<G>(
                &read_results.next().unwrap(),
                j.num_windows,
                j.bucket_width,
            ),
            None => Ok(G::g1_identity()),
        }
    };
    let a = fold_g1(&a_job)?;
    let b1 = fold_g1(&b1_job)?;
    let l = fold_g1(&l_job)?;
    let h = fold_g1(&h_job)?;

    let b2 = if let Some(j) = &b2_job {
        fold_window_sums_g2::<G>(
            &read_results.next().unwrap(),
            j.num_windows,
            j.bucket_width,
        )?
    } else {
        G::g2_identity()
    };

    Ok((a, b1, l, h, b2))
}
