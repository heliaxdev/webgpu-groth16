//! Persistent GPU proving key — pre-uploaded base point buffers.
//!
//! Holds interleaved GLV bases (G1) and direct bases (G2) on the GPU,
//! already converted to Montgomery form. Reused across multiple proofs
//! to eliminate per-proof base uploads and Montgomery conversion.

use crate::gpu::GpuContext;
use crate::gpu::curve::{G1_GPU_BYTES, GpuCurve};

use super::prepared_key::{PreparedProvingKey, interleave_glv_bases};

/// Pre-uploaded GPU base point buffers for a specific circuit.
///
/// Created once per circuit via [`prepare_gpu_proving_key`], then reused
/// across all proofs with [`create_proof_with_gpu_key`](super::create_proof_with_gpu_key).
pub struct GpuProvingKey {
    pub(crate) a_bases_buf: wgpu::Buffer,
    pub(crate) b_g1_bases_buf: wgpu::Buffer,
    pub(crate) l_bases_buf: wgpu::Buffer,
    pub(crate) h_bases_buf: wgpu::Buffer,
    pub(crate) b_g2_bases_buf: wgpu::Buffer,
}

/// Upload proving key bases to the GPU and convert to Montgomery form (one-time cost).
pub fn prepare_gpu_proving_key<G: GpuCurve>(
    ppk: &PreparedProvingKey<G>,
    gpu: &GpuContext<G>,
) -> GpuProvingKey {
    // Interleave G1 bases: [P₀, φ(P₀), P₁, φ(P₁), ...]
    let a_combined = interleave_glv_bases(&ppk.a_bytes, &ppk.a_phi_bytes, G1_GPU_BYTES);
    let b_g1_combined = interleave_glv_bases(&ppk.b_g1_bytes, &ppk.b_g1_phi_bytes, G1_GPU_BYTES);
    let l_combined = interleave_glv_bases(&ppk.l_bytes, &ppk.l_phi_bytes, G1_GPU_BYTES);
    let h_combined = interleave_glv_bases(&ppk.h_bytes, &ppk.h_phi_bytes, G1_GPU_BYTES);

    // Upload to GPU
    let a_bases_buf = gpu.create_storage_buffer("gpk_a_bases", &a_combined);
    let b_g1_bases_buf = gpu.create_storage_buffer("gpk_b1_bases", &b_g1_combined);
    let l_bases_buf = gpu.create_storage_buffer("gpk_l_bases", &l_combined);
    let h_bases_buf = gpu.create_storage_buffer("gpk_h_bases", &h_combined);
    let b_g2_bases_buf = gpu.create_storage_buffer("gpk_b2_bases", &ppk.b_g2_bytes);

    // Convert all bases to Montgomery form on GPU (one-time)
    gpu.convert_to_montgomery(&a_bases_buf, false);
    gpu.convert_to_montgomery(&b_g1_bases_buf, false);
    gpu.convert_to_montgomery(&l_bases_buf, false);
    gpu.convert_to_montgomery(&h_bases_buf, false);
    gpu.convert_to_montgomery(&b_g2_bases_buf, true);

    // Wait for all conversions to complete
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());

    GpuProvingKey {
        a_bases_buf,
        b_g1_bases_buf,
        l_bases_buf,
        h_bases_buf,
        b_g2_bases_buf,
    }
}
