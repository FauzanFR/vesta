use crate::dispatch::{dispatch, Binding, DispatchSpec};
use crate::storage::{alloc_slice, ViewMeta};
use wgpu::util::DeviceExt;
use super::super::Tensor;

impl Tensor {
    /// Binary elementwise — shape A == shape B.
    /// Output shape = shape A.
    pub(crate) fn dispatch_binary(
        &self,
        other: &Tensor,
        shader_name: &str,
    ) -> Tensor {
        assert_eq!(
            self.shape, other.shape,
            "{}: shape mismatch, A={:?}, B={:?}",
            shader_name, self.shape, other.shape
        );

        self.assert_chunk_safe(shader_name);
        other.assert_chunk_safe(shader_name);

        let out_storage = alloc_slice(
            &self.device.device,
            self.element_count,
            self.storage.elem_size(),
            shader_name,
        );
        let out = self.wrap_storage(self.shape.clone(), out_storage);

        for i in 0..self.storage.num_chunks() {
            let chunk_offset = self.storage.chunk_offset(i);
            let chunk_elems  = self.storage.chunk_elements(i);

            let meta_a = ViewMeta::new(&self.shape,  &self.strides,  chunk_offset, chunk_elems);
            let meta_b = ViewMeta::new(&other.shape, &other.strides, chunk_offset, chunk_elems);

            let meta_buf_a = self.device.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("meta_a"),
                    contents: bytemuck::bytes_of(&meta_a),
                    usage: wgpu::BufferUsages::STORAGE,
                }
            );
            let meta_buf_b = self.device.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("meta_b"),
                    contents: bytemuck::bytes_of(&meta_b),
                    usage: wgpu::BufferUsages::STORAGE,
                }
            );

            let bindings = vec![
                Binding { slot: 0, buffer: &self.storage.chunks[i],  read_only: true },
                Binding { slot: 1, buffer: &other.storage.chunks[i], read_only: true },
                Binding { slot: 2, buffer: &out.storage.chunks[i],   read_only: false },
                Binding { slot: 3, buffer: &meta_buf_a,              read_only: true },
                Binding { slot: 4, buffer: &meta_buf_b,              read_only: true },
            ];

            dispatch(&self.device, &DispatchSpec {
                label: shader_name,
                shader_name,
                bindings,
                uniforms: vec![],
                total_elements: chunk_elems,
            });
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Public binary ops
// ---------------------------------------------------------------------------
impl Tensor {
    pub fn add(&self, other: &Tensor) -> Tensor {
        self.dispatch_binary(other, "add")
    }
    pub fn subtract(&self, other: &Tensor) -> Tensor {
        self.dispatch_binary(other, "subtract")
    }
    pub fn multiply(&self, other: &Tensor) -> Tensor {
        self.dispatch_binary(other, "multiply")
    }
    pub fn divide(&self, other: &Tensor) -> Tensor {
        self.dispatch_binary(other, "divide")
    }
    pub fn max(&self, other: &Tensor) -> Tensor {
        self.dispatch_binary(other, "max")
    }
    pub fn min(&self, other: &Tensor) -> Tensor {
        self.dispatch_binary(other, "min")
    }
    pub fn pow(&self, other: &Tensor) -> Tensor {
        self.dispatch_binary(other, "pow")
    }
}