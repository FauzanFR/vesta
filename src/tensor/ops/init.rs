use crate::device::Device;
use crate::storage::alloc_slice;
use crate::tensor::shape::num_elements;
use super::super::Tensor;

impl Tensor {
    pub fn zeros(device: &Device, shape: Vec<usize>) -> Tensor {
        Self::fill(device, shape, 0.0)
    }

    pub fn ones(device: &Device, shape: Vec<usize>) -> Tensor {
        Self::fill(device, shape, 1.0)
    }

    pub fn fill(device: &Device, shape: Vec<usize>, value: f32) -> Tensor {
        let total = num_elements(&shape);
        let storage = alloc_slice(&device.core().device, total, 4, "fill");
        let t = Tensor::wrap_storage_static(device, shape.clone(), storage);
        t.dispatch_init("fill", &[("scalar", bytemuck::bytes_of(&value))]);
        t
    }

    pub fn random(device: &Device, shape: Vec<usize>, seed: Option<u32>) -> Tensor {
        let total = num_elements(&shape);
        let actual_seed: u32 = seed.unwrap_or_else(rand::random);
        let storage = alloc_slice(&device.core().device, total, 4, "random");
        let t = Tensor::wrap_storage_static(device, shape, storage);
        t.dispatch_init("random", &[("seed", bytemuck::bytes_of(&actual_seed))]);
        t
    }

    pub fn arange(device: &Device, shape: Vec<usize>, start: f32, step: f32) -> Tensor {
        let total = num_elements(&shape);
        let storage = alloc_slice(&device.core().device, total, 4, "arange");
        let t = Tensor::wrap_storage_static(device, shape, storage);
        t.dispatch_init("arange", &[
            ("start", bytemuck::bytes_of(&start)),
            ("step",  bytemuck::bytes_of(&step)),
        ]);
        t
    }
}