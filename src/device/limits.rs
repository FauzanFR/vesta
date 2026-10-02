pub fn calc_chunk_size(total: usize, elem_size: usize, device: &wgpu::Device) -> usize {
    let limit_buf  = device.limits().max_buffer_size as usize;
    let limit_bind = device.limits().max_storage_buffer_binding_size as usize;
    let max_bytes  = limit_buf.min(limit_bind);
    let max_elements = max_bytes / elem_size;
    max_elements.min(total)
}