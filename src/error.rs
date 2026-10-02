use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("GPU read failed")]
    GpuRead(#[from] wgpu::BufferAsyncError),

    #[error("device lost")]
    DeviceLost,

    #[error("no GPU adapter found")]
    NoAdapter,
}