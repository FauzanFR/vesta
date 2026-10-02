use crate::dispatch::{dispatch, Binding, DispatchSpec, Uniform};
use crate::storage::{alloc_slice, ViewMeta};
use wgpu::util::DeviceExt;
use super::super::Tensor;

impl Tensor {
    /// Unary elementwise — output shape sama dengan input.
    pub(crate) fn dispatch_unary(
        &self,
        shader_name: &str,
        uniforms: &[(&str, &[u8])],
    ) -> Tensor {
        self.assert_chunk_safe(shader_name);

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

            let meta = ViewMeta::new(&self.shape, &self.strides, chunk_offset, chunk_elems);
            let meta_buf = self.device.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("meta"),
                    contents: bytemuck::bytes_of(&meta),
                    usage: wgpu::BufferUsages::STORAGE,
                }
            );

            let bindings = vec![
                Binding { slot: 0, buffer: &self.storage.chunks[i], read_only: true },
                Binding { slot: 1, buffer: &out.storage.chunks[i], read_only: false },
                Binding { slot: 2, buffer: &meta_buf, read_only: true },
            ];

            let uniform_specs: Vec<Uniform> = uniforms.iter().enumerate()
                .map(|(k, (label, bytes))| Uniform {
                    slot: 3 + k as u32,
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

        out
    }
}

// ---------------------------------------------------------------------------
// Public unary ops
// ---------------------------------------------------------------------------
impl Tensor {
    pub fn relu(&self)    -> Tensor { self.dispatch_unary("relu",    &[]) }
    pub fn sigmoid(&self) -> Tensor { self.dispatch_unary("sigmoid", &[]) }
    pub fn tanh(&self)    -> Tensor { self.dispatch_unary("tanh",    &[]) }
    pub fn exp(&self)     -> Tensor { self.dispatch_unary("exp",     &[]) }
    pub fn log(&self)     -> Tensor { self.dispatch_unary("log",     &[]) }
    pub fn sqrt(&self)    -> Tensor { self.dispatch_unary("sqrt",    &[]) }
    pub fn abs(&self)     -> Tensor { self.dispatch_unary("abs",     &[]) }
    pub fn neg(&self)     -> Tensor { self.dispatch_unary("neg",     &[]) }
    pub fn softplus(&self)-> Tensor { self.dispatch_unary("softplus",&[]) }

    pub fn clip(&self, min: f32, max: f32) -> Tensor {
        self.dispatch_unary("clip", &[
            ("clip_min", bytemuck::bytes_of(&min)),
            ("clip_max", bytemuck::bytes_of(&max)),
        ])
    }

    pub fn leaky_relu(&self, alpha: f32) -> Tensor {
        self.dispatch_unary("leaky_relu", &[
            ("alpha", bytemuck::bytes_of(&alpha))
        ])
    }

    pub fn elu(&self, alpha: f32) -> Tensor {
        self.dispatch_unary("elu", &[
            ("alpha", bytemuck::bytes_of(&alpha))
        ])
    }
}