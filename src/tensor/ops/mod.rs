mod init;
mod unary;
mod binary;
mod ops_impl;
mod reduce;
mod matmul;

use crate::dispatch::{dispatch, Binding, DispatchSpec, Uniform};
use crate::storage::ViewMeta;
use wgpu::util::DeviceExt;
use super::Tensor;

impl Tensor {
    /// Dispatch init — tulis ke buffer sendiri, pakai shader init.
    pub(crate) fn dispatch_init(&self, shader_name: &str, uniforms: &[(&str, &[u8])]) {
        for i in 0..self.storage.num_chunks() {
            let chunk_offset = self.storage.chunk_offset(i);
            let chunk_elems = self.storage.chunk_elements(i);

            let meta = ViewMeta::new(&self.shape, &self.strides, chunk_offset, chunk_elems);
            let meta_buf = self.device.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("meta"),
                    contents: bytemuck::bytes_of(&meta),
                    usage: wgpu::BufferUsages::STORAGE,
                }
            );

            let bindings = vec![
                Binding { slot: 0, buffer: &self.storage.chunks[i], read_only: false },
                Binding { slot: 1, buffer: &meta_buf, read_only: true },
            ];

            let uniform_specs: Vec<Uniform> = uniforms.iter().enumerate()
                .map(|(k, (label, bytes))| Uniform {
                    slot: 2 + k as u32,
                    bytes,
                    label,
                })
                .collect();

            dispatch(&self.device, &DispatchSpec {
                label: shader_name,
                shader_name,
                bindings,
                uniforms: uniform_specs,
                total_elements: chunk_elems,
            });
        }
    }
}