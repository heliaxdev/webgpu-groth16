//! GPU buffer read-back operations.
//!
//! Handles asynchronous GPU → CPU data transfer via staging buffers.

use futures::channel::oneshot;

use super::GpuContext;
use super::curve::GpuCurve;

impl<C: GpuCurve> GpuContext<C> {
    pub async fn read_buffer(
        &self,
        buffer: &wgpu::Buffer,
        size: wgpu::BufferAddress,
    ) -> anyhow::Result<Vec<u8>> {
        let staging_buffer =
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Read Buffer"),
                size,
                usage: wgpu::BufferUsages::MAP_READ
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );
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

    /// Reads multiple GPU buffers in a single command submission for
    /// efficiency.
    ///
    /// All copy commands are batched into one encoder, submitted together, and
    /// then all staging buffers are mapped concurrently. This avoids the
    /// overhead of per-buffer submission and device polling.
    pub async fn read_buffers_batch(
        &self,
        entries: &[(&wgpu::Buffer, wgpu::BufferAddress)],
    ) -> anyhow::Result<Vec<Vec<u8>>> {
        let mut staging = Vec::with_capacity(entries.len());
        for (_, size) in entries {
            staging.push(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Batch Staging Read Buffer"),
                size: *size,
                usage: wgpu::BufferUsages::MAP_READ
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Batch Read Encoder"),
            },
        );
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
