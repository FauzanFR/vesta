struct ViewMeta {
    strides:      array<u32, 8>,
    shape:        array<u32, 8>,
    ndim:         u32,
    total:        u32,
    chunk_offset: u32,
    chunk_size:   u32,
}

@group(0) @binding(0) var<storage, read>       buf_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> buf_out: array<f32>;
@group(0) @binding(2) var<storage, read>       meta_in: ViewMeta;

fn logical_to_physical(logical_flat: u32, m: ViewMeta) -> u32 {
    var remaining = logical_flat;
    var physical  = 0u;
    var default_stride = 1u;
    var default_strides: array<u32, 8>;
    for (var d = i32(m.ndim) - 1; d >= 0; d--) {
        default_strides[d] = default_stride;
        default_stride *= m.shape[d];
    }
    for (var d = 0u; d < m.ndim; d++) {
        let idx   = remaining / default_strides[d];
        remaining = remaining % default_strides[d];
        physical += idx * m.strides[d];
    }
    return physical;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups)        groups:    vec3<u32>,
) {
    let wg_linear   = global_id.y * groups.x + (global_id.x / 256u);
    let local_in_wg = global_id.x % 256u;
    let local_idx   = wg_linear * 256u + local_in_wg;
    if (local_idx >= meta_in.total) { return; }

    let phys = logical_to_physical(local_idx, meta_in);
    let val  = buf_in[phys];

    buf_out[local_idx] = max(val, 0.0);
}