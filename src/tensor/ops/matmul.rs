use crate::dispatch::{dispatch_raw, Binding, Uniform};
use crate::storage::alloc_slice;
use wgpu::util::DeviceExt;
use super::super::Tensor;

impl Tensor {
    /// Matmul: self @ other.
    /// MVP: 2D x 2D, single-chunk. Strides dihormati (lazy transpose works).
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert!(self.shape.len() >= 2,
            "matmul: self minimal 2D, dapat {:?}", self.shape);
        assert_eq!(other.shape.len(), 2,
            "matmul: other harus 2D, dapat {:?}", other.shape);

        // MVP: tanpa batch dims
        assert_eq!(self.shape.len(), 2,
            "matmul: batch dims belum didukung, self={:?}", self.shape);

        // MVP: single chunk
        assert_eq!(self.storage.num_chunks(), 1,
            "matmul: A multi-chunk belum didukung");
        assert_eq!(other.storage.num_chunks(), 1,
            "matmul: B multi-chunk belum didukung");

        let m  = self.shape[0];
        let k  = self.shape[1];
        let k2 = other.shape[0];
        let n  = other.shape[1];

        assert_eq!(k, k2,
            "matmul: inner dim mismatch, A={:?}, B={:?}",
            self.shape, other.shape);

        let out_shape = vec![m, n];
        let out_total = m * n;
        let out_storage = alloc_slice(
            &self.device.device,
            out_total,
            self.storage.elem_size(),
            "matmul",
        );
        // Shader hanya menulis ke chunks[0]; output multi-chunk akan korup diam-diam.
        assert_eq!(out_storage.num_chunks(), 1,
            "matmul: output {}x{} melebihi 1 chunk, belum didukung", m, n);
        let out = self.wrap_storage(out_shape, out_storage);

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Dims {
            m: u32, k: u32, n: u32,
            batch_offset_a: u32,
            batch_offset_b: u32,
            batch_offset_c: u32,
            stride_a0: u32, stride_a1: u32,
            stride_b0: u32, stride_b1: u32,
        }

        let dims = Dims {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            batch_offset_a: 0,
            batch_offset_b: 0,
            batch_offset_c: 0,
            stride_a0: self.strides[0] as u32,
            stride_a1: self.strides[1] as u32,
            stride_b0: other.strides[0] as u32,
            stride_b1: other.strides[1] as u32,
        };

        let _dims_buf = self.device.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("matmul_dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            }
        );

        let bindings = vec![
            Binding { slot: 0, buffer: &self.storage.chunks[0],  read_only: true },
            Binding { slot: 1, buffer: &other.storage.chunks[0], read_only: true },
            Binding { slot: 2, buffer: &out.storage.chunks[0],   read_only: false },
        ];
        let uniforms = vec![
            Uniform { slot: 3, bytes: bytemuck::bytes_of(&dims), label: "dims" },
        ];

        let wg_x = (n as u32 + 15) / 16;
        let wg_y = (m as u32 + 15) / 16;

        dispatch_raw(&self.device, "matmul", "matmul", &bindings, &uniforms, (wg_x, wg_y, 1));

        out
    }
}