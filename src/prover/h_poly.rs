//! H polynomial computation via GPU.
//!
//! The quotient polynomial H(x) satisfies A(x)·B(x) − C(x) = H(x)·Z(x) where
//! Z(x) is the vanishing polynomial over the constraint domain. Computing H
//! requires FFT-based polynomial arithmetic performed entirely on the GPU.
//!
//! CPU pre-computes constant factors (twiddle factors, coset shifts, Z⁻¹) and
//! uploads them alongside the constraint evaluations. The GPU then runs the
//! full pipeline in a single command buffer:
//!
//!   toMont → iNTT → cosetShift → NTT → pointwise(H=(A·B−C)/Z) → iNTT →
//! fromMont

use anyhow::Result;
use ff::{Field, PrimeField};

use super::marshal_scalars;
use crate::gpu::curve::GpuCurve;
use crate::gpu::{GpuContext, HPolyBuffers};

pub(crate) struct HPolyPending {
    pub h_buf: wgpu::Buffer,
    pub n: usize,
}

/// Submit the H polynomial pipeline to the GPU (non-blocking).
/// Returns a pending handle that can be read later with `read_h_poly_result`.
pub(crate) fn submit_h_poly<G: GpuCurve>(
    gpu: &GpuContext<G>,
    a_values: &[G::Scalar],
    b_values: &[G::Scalar],
    c_values: &[G::Scalar],
) -> Result<HPolyPending> {
    let n = a_values.len().next_power_of_two();

    // CPU pre-computes constant factors for the coset-FFT pipeline:
    // - omega_n: primitive n-th root of unity in Fr
    // - n_inv: 1/n for normalizing iNTT output
    // - coset_generator: multiplicative generator g for coset evaluation
    // - shifts[i] = g^i / n (combined coset shift + iNTT normalization)
    // - z_inv = 1/(g^n - 1) for dividing by the vanishing polynomial
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

    // Zero-pad constraint evaluations to next power of two.
    let mut a_coeffs = a_values.to_vec();
    a_coeffs.resize(n, G::Scalar::ZERO);
    let mut b_coeffs = b_values.to_vec();
    b_coeffs.resize(n, G::Scalar::ZERO);
    let mut c_coeffs = c_values.to_vec();
    c_coeffs.resize(n, G::Scalar::ZERO);

    // Upload all buffers to VRAM.
    let a_buf =
        gpu.create_storage_buffer("A", &marshal_scalars::<G>(&a_coeffs));
    let b_buf =
        gpu.create_storage_buffer("B", &marshal_scalars::<G>(&b_coeffs));
    let c_buf =
        gpu.create_storage_buffer("C", &marshal_scalars::<G>(&c_coeffs));
    let h_buf = gpu.create_empty_buffer("H", (n * 32) as u64);

    let tw_inv_n_buf = gpu.create_storage_buffer(
        "TwInvN",
        &marshal_scalars::<G>(&inv_twiddles_n),
    );
    let tw_fwd_n_buf = gpu.create_storage_buffer(
        "TwFwdN",
        &marshal_scalars::<G>(&fwd_twiddles_n),
    );
    let shifts_buf =
        gpu.create_storage_buffer("Shifts", &marshal_scalars::<G>(&shifts));
    let inv_shifts_buf = gpu
        .create_storage_buffer("InvShifts", &marshal_scalars::<G>(&inv_shifts));
    let z_invs_buf =
        gpu.create_storage_buffer("ZInvs", &marshal_scalars::<G>(&z_invs));

    // Dispatch the full H pipeline on GPU using a single command buffer.
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

/// Read the result of a previously submitted H polynomial pipeline.
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
