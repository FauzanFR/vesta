use std::sync::Arc;
use wgpu::util::DeviceExt;
use crate::{device::limits::calc_chunk_size, storage::TensorSlice};

pub fn alloc_slice(device: &wgpu::Device, total: usize, elem_size: usize, label: &str) -> TensorSlice {
    
    if total == 0 {
        return TensorSlice {
            chunks: Arc::new(Vec::new()),
            chunk_size: 0,   // atau 0, tidak dipakai
            total: 0,
            elem_size,
        };
    }

    let chunk_size = calc_chunk_size(total, elem_size, device);
    let num_chunks = total.div_ceil(chunk_size);

    let chunks: Vec<wgpu::Buffer> = (0..num_chunks)
        .map(|i| {
            let size = chunk_size.min(total - i * chunk_size);
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{}_{}", label, i)),
                size: (size * elem_size) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .collect();

    TensorSlice {
        chunks: Arc::new(chunks),
        chunk_size,
        total,
        elem_size
    }
}

pub fn upload_slice(device: &wgpu::Device, data: &[f32], label: &str) -> TensorSlice {
    // khusus f32, elem_size = 4
    let elem_size = 4;
    let total = data.len();
    let chunk_size = calc_chunk_size(total, elem_size, device);

    let chunks: Vec<wgpu::Buffer> = data
        .chunks(chunk_size)
        .enumerate()
        .map(|(i, slice)| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{}_{}", label, i)),
                contents: bytemuck::cast_slice(slice),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        })
        .collect();

    TensorSlice {
        chunks: Arc::new(chunks),
        chunk_size,
        total,
        elem_size
    }
}