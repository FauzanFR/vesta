use crate::error::TensorError;

pub async fn request_device() -> Result<(wgpu::Device, wgpu::Queue), TensorError> {
    // Backend dipilih otomatis. Paksa lewat env var bila perlu:
    //   WGPU_BACKEND=dx12 | vulkan | metal | gl
    // (Vulkan di beberapa iGPU bermasalah; workaround: WGPU_BACKEND=dx12)
    let instance = wgpu::Instance::new(
        wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
    );

    let adapter = instance
        .request_adapter(&Default::default())
        .await
        .map_err(|_| TensorError::NoAdapter)?;

    let _ = adapter.features();

    let (device, queue) = adapter
        .request_device(&Default::default())
        .await
        .map_err(|_| TensorError::DeviceLost)?;
    Ok((device, queue))
}
