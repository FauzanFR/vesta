#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewMeta {
    pub strides: [[u32; 4]; 2],
    pub shape: [[u32; 4]; 2],
    pub ndim: u32,
    pub total: u32,
    pub chunk_start: u32,
    pub valid_count: u32,
    pub _padding: [u32; 2],
}

impl ViewMeta {
    pub fn new(shape: &[usize], strides: &[usize], chunk_start: usize, valid_count: usize) -> Self {
        let ndim = shape.len();
        assert!(ndim <= 8, "ViewMeta: max 8 dims, got {}", ndim);

        let mut s = [[0u32; 4]; 2];
        let mut st = [[0u32; 4]; 2];
        for i in 0..ndim {
            s[i / 4][i % 4] = shape[i] as u32;
            st[i / 4][i % 4] = strides[i] as u32;
        }

        ViewMeta {
            shape: s,
            strides: st,
            ndim: ndim as u32,
            total: shape.iter().product::<usize>() as u32,
            chunk_start: chunk_start as u32,
            valid_count: valid_count as u32,
            _padding: [0u32; 2],
        }
    }
}