pub struct Binding<'a> {
    pub slot: u32,
    pub buffer: &'a wgpu::Buffer,
    pub read_only: bool,
}

pub struct Uniform<'a> {
    pub slot: u32,
    pub bytes: &'a [u8],
    pub label: &'a str,
}

pub struct DispatchSpec<'a> {
    pub label: &'a str,
    pub shader_name: &'a str,
    pub bindings: Vec<Binding<'a>>,
    pub uniforms: Vec<Uniform<'a>>,
    pub total_elements: usize,
}