struct ViewMeta {
    strides:      array<u32, 8>,
    shape:        array<u32, 8>,
    ndim:         u32,
    total:        u32,
    chunk_offset: u32,
    chunk_size:   u32,
}

@group(0) @binding(0)var<storage, read_write>   a: array<f32>;
@group(0) @binding(1)var<storage, read>         meta_a: ViewMeta;
@group(0) @binding(2)var<uniform>               start: f32;
@group(0) @binding(3)var<uniform>               step: f32;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>
) {
    let wg_linear   = global_id.y * groups.x + (global_id.x / 256u);
    let local_in_wg = global_id.x % 256u;
    let local_idx   = wg_linear * 256u + local_in_wg;
    if (local_idx >= meta_a.chunk_size) { return; }

    let logical_flat = meta_a.chunk_offset + local_idx;
    if (logical_flat >= meta_a.total) { return; }
    
    a[local_idx] = f32(logical_flat) * step + start;
}