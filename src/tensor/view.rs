use super::Tensor;

impl Tensor {
    /// Lazy transpose — swap shape + strides, zero-copy.
    #[allow(non_snake_case)]
    pub fn T(&self) -> Tensor {
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.reverse();
        new_strides.reverse();

        Tensor {
            device: self.device.clone(),
            vram_id: None,          // view — tidak own VRAM
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            element_count: self.element_count,
            storage: self.storage.clone(),
        }
    }
}