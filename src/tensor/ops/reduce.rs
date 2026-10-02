use std::sync::mpsc;
use crate::Device;
use crate::dispatch::{Binding, DispatchSpec, Uniform, dispatch};
use crate::storage::{ViewMeta, alloc_slice};
use crate::tensor::{calculate_strides, num_elements};
use wgpu;
use wgpu::util::DeviceExt;
use super::super::Tensor;

impl Tensor {
    /// Reduce satu chunk ke scalar f32 via multi-pass GPU.
    fn reduce_chunk(&self, chunk_idx: usize) -> f32 {
        let chunk_elems  = self.storage.chunk_elements(chunk_idx);
        let chunk_offset = self.storage.chunk_offset(chunk_idx);

        let intermediate_size = chunk_elems.div_ceil(256).max(1);
        let bytes = (intermediate_size * self.storage.elem_size()) as u64;

        // Ping-pong buffer
        let buf_a = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reduce_a"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_b = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reduce_b"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pass 1 — chunk → buf_a
        {
            let meta = ViewMeta::new(&self.shape, &self.strides, chunk_offset, chunk_elems);
            let meta_buf = self.device.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("meta_r1"),
                    contents: bytemuck::bytes_of(&meta),
                    usage: wgpu::BufferUsages::STORAGE,
                }
            );

            dispatch(&self.device, &DispatchSpec {
                label: "reduce_sum",
                shader_name: "reduce_sum",
                bindings: vec![
                    Binding { slot: 0, buffer: &self.storage.chunks[chunk_idx], read_only: true },
                    Binding { slot: 1, buffer: &buf_a, read_only: false },
                    Binding { slot: 2, buffer: &meta_buf, read_only: true },
                ],
                uniforms: vec![],
                total_elements: chunk_elems,
            });
        }

        // Multi-pass ping-pong sampai 1 elemen
        let mut remaining = intermediate_size;
        let mut read_a    = true;

        while remaining > 1 {
            let (input, output) = if read_a {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };

            let flat_shape   = vec![remaining];
            let flat_strides = vec![1usize];
            let meta = ViewMeta::new(&flat_shape, &flat_strides, 0, remaining);
            let meta_buf = self.device.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("meta_r2"),
                    contents: bytemuck::bytes_of(&meta),
                    usage: wgpu::BufferUsages::STORAGE,
                }
            );

            dispatch(&self.device, &DispatchSpec {
                label: "reduce_sum",
                shader_name: "reduce_sum",
                bindings: vec![
                    Binding { slot: 0, buffer: input, read_only: true },
                    Binding { slot: 1, buffer: output, read_only: false },
                    Binding { slot: 2, buffer: &meta_buf, read_only: true },
                ],
                uniforms: vec![],
                total_elements: remaining,
            });

            remaining = remaining.div_ceil(256);
            read_a    = !read_a;
        }

        // Baca 1 f32
        let final_buf = if read_a { &buf_a } else { &buf_b };
        let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reduce_staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(final_buf, 0, &staging, 0, 4);
        self.device.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| { tx.send(v).unwrap(); });
        self.device.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().unwrap().unwrap();

        let data   = slice.get_mapped_range().unwrap();
        let result = bytemuck::cast_slice::<u8, f32>(&data[..])[0];
        drop(data);
        staging.unmap();
        result
    }

    /// Global sum — return scalar f32.
    pub fn sum(&self) -> f32 {
        let mut total = 0.0f32;
        for i in 0..self.storage.num_chunks() {
            total += self.reduce_chunk(i);
        }
        total
    }

    /// Global mean — sum / element_count.
    pub fn mean(&self) -> f32 {
        self.sum() / self.element_count as f32
    }

    pub fn sum_axis(&self, axis: usize) -> Tensor {
        assert!(
            axis < self.shape.len(),
            "sum_axis: axis {} out of range for shape {:?}",
            axis, self.shape
        );

        let out_shape: Vec<usize> = self.shape.iter().enumerate()
            .filter(|(d, _)| *d != axis)
            .map(|(_, &s)| s)
            .collect();
        let out_total = num_elements(&out_shape);

        // Multi-chunk → fallback CPU (benar tapi lambat). Shader reduce_axis belum
        // chunk-aware: indeks fisik global dipakai langsung dan output tertimpa
        // per chunk. Single chunk → GPU langsung.
        if self.storage.num_chunks() > 1 {
            return self.sum_axis_cpu(axis, out_shape, out_total);
        }
        self.sum_axis_direct(axis, out_shape, out_total)
    }

    fn sum_axis_direct(&self, axis: usize, out_shape: Vec<usize>, out_total: usize) -> Tensor {
        let out_storage = alloc_slice(
            &self.device.device,
            out_total,
            self.storage.elem_size(),
            "sum_axis",
        );
        let out = self.wrap_storage(out_shape.clone(), out_storage);
        let out_strides = out.strides.clone();

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct AxisInfo { axis: u32, axis_size: u32 }

        let _ax_buf = self.device.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("axis_info"),
                contents: bytemuck::bytes_of(&AxisInfo {
                    axis: axis as u32,
                    axis_size: self.shape[axis] as u32,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            }
        );

        for i in 0..self.storage.num_chunks() {
            let chunk_offset = self.storage.chunk_offset(i);
            let chunk_elems  = self.storage.chunk_elements(i);

            let meta_in = ViewMeta::new(&self.shape, &self.strides, chunk_offset, chunk_elems);
            let meta_out = ViewMeta::new(&out_shape, &out_strides, 0, out_total);

            let meta_in_buf = self.device.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("meta_in"),
                    contents: bytemuck::bytes_of(&meta_in),
                    usage: wgpu::BufferUsages::STORAGE,
                }
            );
            let meta_out_buf = self.device.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("meta_out"),
                    contents: bytemuck::bytes_of(&meta_out),
                    usage: wgpu::BufferUsages::STORAGE,
                }
            );

            dispatch(&self.device, &DispatchSpec {
                label: "sum_axis",
                shader_name: "reduce_axis",
                bindings: vec![
                    Binding { slot: 0, buffer: &self.storage.chunks[i], read_only: true },
                    Binding { slot: 1, buffer: &out.storage.chunks[0], read_only: false },
                    Binding { slot: 2, buffer: &meta_in_buf, read_only: true },
                    Binding { slot: 3, buffer: &meta_out_buf, read_only: true },
                ],
                uniforms: vec![
                    Uniform { slot: 4, bytes: bytemuck::bytes_of(&AxisInfo {
                        axis: axis as u32,
                        axis_size: self.shape[axis] as u32,
                    }), label: "axis_info" },
                ],
                total_elements: out_total,
            });
        }

        out
    }

    /// Fallback: baca ke CPU, akumulasi, upload balik.
    /// Untuk multi-chunk + axis=0 di mana output element bisa dapat
    /// kontribusi dari beberapa chunk.
    fn sum_axis_cpu(&self, axis: usize, out_shape: Vec<usize>, out_total: usize) -> Tensor {
        let data = self.to_cpu().unwrap();
        let in_strides  = calculate_strides(&self.shape);
        let out_strides = calculate_strides(&out_shape);

        let mut out_data = vec![0.0f32; out_total];
        for in_flat in 0..self.element_count {
            let mut remaining = in_flat;
            let mut out_flat  = 0;
            let mut out_dim   = 0;
            for d in 0..self.shape.len() {
                let idx = remaining / in_strides[d];
                remaining %= in_strides[d];
                if d != axis {
                    out_flat += idx * out_strides[out_dim];
                    out_dim += 1;
                }
            }
            out_data[out_flat] += data[in_flat];
        }

        let device = Device::from_core(self.device.clone());
        Tensor::from_vec(&device, out_shape, &out_data)
    }

}