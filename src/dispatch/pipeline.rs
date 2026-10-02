use std::sync::Arc;
use crate::device::DeviceCore;
use crate::shaders::get_shader;

pub fn get_shader_module(core: &DeviceCore, name: &str) -> Arc<wgpu::ShaderModule> {
    let mut caches = core.caches.lock().unwrap();
    if let Some(m) = caches.shaders.get(name) {
        return m.clone();
    }
    let module = Arc::new(core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(get_shader(name).into()),
    }));
    caches.shaders.insert(name.to_string(), module.clone());
    module
}

pub fn get_layout(
    core: &DeviceCore,
    entries: &[wgpu::BindGroupLayoutEntry],
    hash: u64,
) -> Arc<wgpu::BindGroupLayout> {
    let mut caches = core.caches.lock().unwrap();
    if let Some(l) = caches.layouts.get(&hash) {
        return l.clone();
    }
    let layout = Arc::new(core.device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label: Some("layout"),
            entries,
        },
    ));
    caches.layouts.insert(hash, layout.clone());
    layout
}

pub fn get_pipeline(
    core: &DeviceCore,
    shader_name: &str,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
    hash: u64,
) -> Arc<wgpu::ComputePipeline> {
    let key = (shader_name.to_string(), hash);
    let mut caches = core.caches.lock().unwrap();
    if let Some(p) = caches.pipelines.get(&key) {
        return p.clone();
    }
    let pl = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(shader_name),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    let pipeline = Arc::new(core.device.create_compute_pipeline(
        &wgpu::ComputePipelineDescriptor {
            label: Some(shader_name),
            layout: Some(&pl),
            module: shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        },
    ));
    caches.pipelines.insert(key, pipeline.clone());
    pipeline
}