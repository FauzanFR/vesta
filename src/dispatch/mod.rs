mod layout;
mod pipeline;
mod spec;

pub use spec::{Binding, Uniform, DispatchSpec};

use wgpu::util::DeviceExt;
use crate::device::DeviceCore;

pub fn dispatch(core: &DeviceCore, spec: &DispatchSpec) {
    let wg = (spec.total_elements as u32 + 255) / 256;
    let grid = if wg <= 65535 {
        (wg, 1, 1)
    } else {
        (65535, (wg + 65534) / 65535, 1)
    };

    dispatch_raw(core, spec.label, spec.shader_name, &spec.bindings, &spec.uniforms, grid);
}

pub(crate) fn dispatch_raw(
    core: &DeviceCore,
    label: &str,
    shader_name: &str,
    bindings: &[Binding],
    uniforms: &[Uniform],
    grid: (u32, u32, u32),
) {

    // 1. Bangun uniform buffers (dipegang selama dispatch)
    let uniform_buffers: Vec<wgpu::Buffer> = uniforms.iter()
        .map(|u| core.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(u.label),
            contents: u.bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        }))
        .collect();

    // 2. Bangun layout entries + hash
    let layout_entries = layout::build_layout_entries(&bindings, &uniforms);
    let hash = layout::layout_hash(&layout_entries);

    // 3. Ambil (atau buat) shader, layout, pipeline dari cache
    let shader = pipeline::get_shader_module(core, shader_name);
    let bgl = pipeline::get_layout(core, &layout_entries, hash);
    let pipeline = pipeline::get_pipeline(core, shader_name, &shader, &bgl, hash);

    // 4. Bangun bind group
    let mut bind_entries: Vec<wgpu::BindGroupEntry> = bindings.iter()
        .map(|b| layout::binding_entry(b.slot, b.buffer))
        .collect();
    for (i, u) in uniforms.iter().enumerate() {
        bind_entries.push(layout::binding_entry(u.slot, &uniform_buffers[i]));
    }
    bind_entries.sort_by_key(|e| e.binding);

    let bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &bgl,
        entries: &bind_entries,
    });

    // 5. Encode + submit

    let mut encoder = core.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some(label) }
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(grid.0, grid.1, grid.2);
    }
    core.queue.submit(Some(encoder.finish()));
}