mod shape;
mod view;
pub(crate) mod ops;

pub use shape::{calculate_strides, logical_to_physical, num_elements};

use std::sync::Arc;

use crate::device::{Device, DeviceCore};
use crate::error::TensorError;
use crate::storage::{upload_slice, TensorSlice};

pub struct  Tensor {
    pub(crate) device: Arc<DeviceCore>,
    pub(crate) vram_id: Option<usize>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
    pub(crate) element_count: usize,
    pub(crate) storage: TensorSlice,
}

impl Tensor {
    /// Buat tensor dari data CPU. Data di-upload ke GPU.
    pub fn from_vec(device: &Device, shape: Vec<usize>, data: &[f32]) -> Self {
        let total = num_elements(&shape);
        assert_eq!(
            total, data.len(),
            "from_vec: shape {:?} butuh {} elemen, dapat {}",
            shape, total, data.len()
        );

        let core = device.core();
        let storage = upload_slice(&core.device, data, "from_vec");
        let bytes = total * storage.elem_size();
        let vram_id = core.state.lock().unwrap().register(bytes);

        Self {
            device: core.clone(),
            vram_id: Some(vram_id),
            strides: calculate_strides(&shape),
            shape,
            offset: 0,
            element_count: total,
            storage,
        }
    }

    pub(crate) fn wrap_storage_static(
        device: &Device,
        shape: Vec<usize>,
        storage: TensorSlice,
    ) -> Tensor {
        let total = num_elements(&shape);
        let bytes = total * storage.elem_size();
        let vram_id = device.core().state.lock().unwrap().register(bytes);

        Tensor {
            device: device.core().clone(),
            vram_id: Some(vram_id),
            strides: calculate_strides(&shape),
            shape,
            offset: 0,
            element_count: total,
            storage,
        }
    }
    
    pub(crate) fn wrap_storage(&self, shape: Vec<usize>, storage: TensorSlice) -> Tensor {
        let total = num_elements(&shape);
        let bytes = total * storage.elem_size();
        let vram_id = self.device.state.lock().unwrap().register(bytes);

        Tensor {
            device: self.device.clone(),
            vram_id: Some(vram_id),
            strides: calculate_strides(&shape),
            shape,
            offset: 0,
            element_count: total,
            storage,
        }
    }

        pub fn mean_axis(&self, axis: usize) -> Tensor {
        assert!(axis < self.shape.len(),
            "mean_axis: axis {} out of range for shape {:?}", axis, self.shape);
        let axis_size = self.shape[axis] as f32;
        let summed = self.sum_axis(axis);
        self.scalar_div(&summed, axis_size)
    }

    /// Bagi tensor dengan scalar. Helper untuk mean_axis.
    fn scalar_div(&self, t: &Tensor, divisor: f32) -> Tensor {
        let _scalar = Tensor::from_vec(
            &Device::from_core(self.device.clone()),
            vec![1],
            &[divisor],
        );
        // broadcast: karena semua elemen dibagi angka sama,
        // kita bisa pakai shader divide dengan scalar di-ulang
        // Tapi cara paling mudah: bikin tensor divisor dengan shape sama
        let divisor_data = vec![divisor; t.element_count];
        let divisor_t = Tensor::from_vec(
            &Device::from_core(self.device.clone()),
            t.shape.clone(),
            &divisor_data,
        );
        t.divide(&divisor_t)
    }


    /// True kalau layout memori sama dengan layout logis (bukan view hasil T()).
    pub(crate) fn is_contiguous(&self) -> bool {
        self.offset == 0 && self.strides == calculate_strides(&self.shape)
    }

    /// Guard: op elementwise multi-chunk belum benar untuk view non-contiguous
    /// (indeks fisik bisa jatuh di chunk lain). Gagal keras daripada hasil salah diam-diam.
    pub(crate) fn assert_chunk_safe(&self, op: &str) {
        assert!(
            self.storage.num_chunks() <= 1 || self.is_contiguous(),
            "{}: tensor non-contiguous (mis. hasil T()) dengan multi-chunk belum didukung",
            op
        );
    }

    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn strides(&self) -> &[usize] { &self.strides }
    pub fn element_count(&self) -> usize { self.element_count }
    pub fn offset(&self) -> usize { self.offset }

    /// Baca tensor dari GPU ke CPU.
    /// Untuk view non-contiguous (hasil T()), data di-reorder ke logical order.
    pub fn to_cpu(&self) -> Result<Vec<f32>, TensorError> {
        let mut raw: Vec<f32> = Vec::with_capacity(self.element_count);

        for i in 0..self.storage.num_chunks() {
            let byte_size = self.storage.chunk_bytes(i);
            let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: byte_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder = self.device.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(
                &self.storage.chunks[i], 0,
                &staging, 0,
                byte_size,
            );
            self.device.queue.submit(Some(encoder.finish()));

            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |v| { tx.send(v).unwrap(); });
            self.device.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            rx.recv().unwrap()?;

            let data = slice.get_mapped_range().unwrap();
            raw.extend_from_slice(bytemuck::cast_slice(&data));
            drop(data);
            staging.unmap();
        }

        // Contiguous — langsung return
        let default_strides = calculate_strides(&self.shape);
        if self.strides == default_strides && self.offset == 0 {
            return Ok(raw);
        }

        // Non-contiguous — walk logical indices, reorder
        let mut result = vec![0.0f32; self.element_count];
        for logical_flat in 0..self.element_count {
            let physical = logical_to_physical(logical_flat, &self.shape, &self.strides)
                + self.offset;
            result[logical_flat] = raw[physical];
        }
        Ok(result)
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if let Some(id) = self.vram_id {
            self.device.state.lock().unwrap().free(id);
        }
    }
}