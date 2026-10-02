use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct TensorSlice {
    pub chunks: Arc<Vec<wgpu::Buffer>>,
    pub(crate) chunk_size: usize,
    pub(crate) total: usize,
    pub(crate) elem_size: usize,
}

impl TensorSlice {
    pub fn num_chunks(&self) -> usize { self.chunks.len() }
    
    pub fn chunk_elements(&self, i: usize) -> usize {
        let start = i * self.chunk_size;
        (start + self.chunk_size).min(self.total) - start
    }

    pub fn chunk_bytes(&self, i: usize) -> u64 {
        (self.chunk_elements(i) * self.elem_size) as u64
    }

    pub fn chunk_offset(&self, i: usize) -> usize {
        i * self.chunk_size
    }

    pub fn elem_size(&self) -> usize { self.elem_size }
    pub fn total(&self) -> usize { self.total }
    pub fn chunk_size(&self) -> usize { self.chunk_size }
}