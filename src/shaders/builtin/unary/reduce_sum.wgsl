struct ViewMeta {
    strides:      array<u32, 8>,
    shape:        array<u32, 8>,
    ndim:         u32,
    total:        u32,
    chunk_offset: u32,
    chunk_size:   u32,
    _padding:     vec2<u32>,
}

@group(0) @binding(0) var<storage, read>       buf_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> buf_out: array<f32>;
@group(0) @binding(2) var<storage, read>       meta_in: ViewMeta;

var<workgroup> shared_mem: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id)  global_id:  vec3<u32>,
    @builtin(local_invocation_id)   local_id:   vec3<u32>,
    @builtin(workgroup_id)          wg_id:      vec3<u32>,
    @builtin(num_workgroups)        groups:     vec3<u32>,
) {
    let wg_linear   = wg_id.y * groups.x + wg_id.x;
    let local_idx   = wg_linear * 256u + local_id.x;
    let global_idx  = meta_in.chunk_offset + local_idx;

    // Load ke shared_mem memory — thread di luar range load 0 (identity untuk sum)
    if (global_idx < meta_in.chunk_offset + meta_in.chunk_size && global_idx < meta_in.total) {
        shared_mem[local_id.x] = buf_in[local_idx];
    } else {
        shared_mem[local_id.x] = 0.0;
    }

    workgroupBarrier();

    // Reduction tree dalam workgroup
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (local_id.x < stride) {
            shared_mem[local_id.x] += shared_mem[local_id.x + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // Thread 0 tiap workgroup tulis hasil ke buf_out
    if (local_id.x == 0u) {
        buf_out[wg_linear] = shared_mem[0];
    }
}
