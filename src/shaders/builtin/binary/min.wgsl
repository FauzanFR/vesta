struct ViewMeta {
    strides:      array<u32, 8>,
    shape:        array<u32, 8>,
    ndim:         u32,
    total:        u32,
    chunk_offset: u32,
    chunk_size:   u32,
}

@group(0) @binding(0) var<storage, read>        buf_a:  array<f32>;
@group(0) @binding(1) var<storage, read>        buf_b:  array<f32>;
@group(0) @binding(2) var<storage, read_write>  buf_out: array<f32>;
@group(0) @binding(3) var<storage, read>        meta_a: ViewMeta;
@group(0) @binding(4) var<storage, read>        meta_b: ViewMeta;

fn logical_to_physical(logical_flat: u32, metadata: ViewMeta) -> u32 {
    var remaining = logical_flat;
    var physical  = 0u;

    var default_stride = 1u;
    var default_strides: array<u32, 8>;
    for (var d = i32(metadata.ndim) - 1; d >= 0; d--) {
        default_strides[d] = default_stride;
        default_stride *= metadata.shape[d];
    }

    for (var d = 0u; d < metadata.ndim; d++) {
        let idx  = remaining / default_strides[d];
        remaining = remaining % default_strides[d];
        physical += idx * metadata.strides[d];
    }
    return physical;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32> 
) {
    let wg_linear   = global_id.y * groups.x + (global_id.x / 256u);
    let local_in_wg = global_id.x % 256u;
    let local_idx   = wg_linear * 256u + local_in_wg;
    if (local_idx >= meta_a.chunk_size) { return; }

    // logical flat = posisi global element dalam tensor
    let logical_flat = meta_a.chunk_offset + local_idx;
    if (logical_flat >= meta_a.total) { return; }

    let phys_a = logical_to_physical(logical_flat, meta_a);
    let phys_b = logical_to_physical(logical_flat, meta_b);

    // phys_a/phys_b adalah global physical index
    // tapi buf_a/buf_b hanya berisi chunk ini → kurangi chunk_offset
    buf_out[local_idx] = min(buf_a[phys_a - meta_a.chunk_offset], buf_b[phys_b - meta_b.chunk_offset]);
}
