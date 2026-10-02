pub fn storage_layout(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn uniform_layout(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn binding_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// Bangun layout entries dari bindings + uniforms, di-sort by slot.
pub fn build_layout_entries(
    bindings: &[super::spec::Binding],
    uniforms: &[super::spec::Uniform],
) -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut entries: Vec<_> = bindings.iter()
        .map(|b| storage_layout(b.slot, b.read_only))
        .chain(uniforms.iter().map(|u| uniform_layout(u.slot)))
        .collect();
    entries.sort_by_key(|e| e.binding);
    entries
}

/// Hash struktur layout — untuk cache key.
pub fn layout_hash(entries: &[wgpu::BindGroupLayoutEntry]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut h = DefaultHasher::new();
    for e in entries {
        e.binding.hash(&mut h);
        match &e.ty {
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                ..
            } => {
                0u8.hash(&mut h);
                read_only.hash(&mut h);
            }
            wgpu::BindingType::Buffer { .. } => {
                1u8.hash(&mut h);
            }
            _ => {
                2u8.hash(&mut h);
            }
        }
    }
    h.finish()
}