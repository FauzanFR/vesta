mod init;
pub mod state;
pub mod limits;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::device::state::DeviceState;
use crate::device::init::request_device;
use crate::error::TensorError;

pub struct GpuCaches {
    pub shaders: HashMap<String, Arc<wgpu::ShaderModule>>,
    pub layouts: HashMap<u64, Arc<wgpu::BindGroupLayout>>,
    pub pipelines: HashMap<(String, u64), Arc<wgpu::ComputePipeline>>,
}

pub struct DeviceCore {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub state: Mutex<DeviceState>,
    pub caches: Mutex<GpuCaches>,
}

pub struct Device {
    core: Arc<DeviceCore>,
}

impl Device {
    pub async fn init() -> Result<Self, TensorError> {
        let (device, queue) = request_device().await?;
        let state = DeviceState::new();

        let caches = GpuCaches {
            shaders: HashMap::new(),
            layouts: HashMap::new(),
            pipelines: HashMap::new(),
        };

        let core = DeviceCore {
            device,
            queue,
            state: Mutex::new(state),
            caches: Mutex::new(caches),
        };
        Ok(Self { core: Arc::new(core) })
    }

    pub fn new() -> Result<Self, TensorError> {
        pollster::block_on(Self::init())
    }

    pub(crate) fn from_core(core: Arc<DeviceCore>) -> Self { Self { core } }
    pub fn core(&self) -> &Arc<DeviceCore> { &self.core }
    pub fn vram_used(&self) -> usize { self.core.state.lock().unwrap().vram_used() }
    pub fn tensor_count(&self) -> usize { self.core.state.lock().unwrap().tensor_count }
    pub fn limits(&self) -> wgpu::Limits { self.core.device.limits() }
}