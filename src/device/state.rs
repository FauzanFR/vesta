use std::collections::HashMap;

pub struct DeviceState {
    pub tensor_count: usize,
    pub vram_used: usize,           // running total
    pub vram_registry: HashMap<usize, usize>,
    next_id: usize,
}

impl DeviceState {
    pub fn new() -> Self {
        Self {
            tensor_count: 0,
            vram_used: 0,
            vram_registry: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn register(&mut self, bytes: usize) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.tensor_count += 1;
        self.vram_used += bytes;
        self.vram_registry.insert(id, bytes);
        id
    }

    pub fn free(&mut self, id: usize) {
        if let Some(bytes) = self.vram_registry.remove(&id) {
            self.tensor_count -= 1;
            self.vram_used = self.vram_used.saturating_sub(bytes);
        }
    }

    pub fn vram_used(&self) -> usize { self.vram_used }
}