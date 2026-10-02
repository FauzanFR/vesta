// Per-axis reduction shader
// Tiap thread handle satu elemen output
// Loop sepanjang axis yang di-sum, akumulasi ke output
//
// Contoh shape [3,4,5], axis=1:
//   output shape [3,5], output total = 15
//   tiap thread i → loop k=0..4, sum buf_in[...] dengan k di axis=1
//
// Bindings:
//   0: buf_in   (read)
//   1: buf_out  (write)
//   2: meta_in  (shape/strides tensor input)
//   3: meta_out (shape/strides tensor output)
//   4: axis_info (uniform: axis + axis_size)

struct ViewMeta {
    strides:      array<u32, 8>,
    shape:        array<u32, 8>,
    ndim:         u32,
    total:        u32,
    chunk_offset: u32,
    chunk_size:   u32,
    _padding:     vec2<u32>,
}

struct AxisInfo {
    axis:      u32,
    axis_size: u32,
}

@group(0) @binding(0) var<storage, read>       buf_in:   array<f32>;
@group(0) @binding(1) var<storage, read_write> buf_out:  array<f32>;
@group(0) @binding(2) var<storage, read>       meta_in:  ViewMeta;
@group(0) @binding(3) var<storage, read>       meta_out: ViewMeta;
@group(0) @binding(4) var<uniform>             ax:       AxisInfo;

// Hitung physical index dari multi-dim index + strides
fn to_physical(indices: array<u32, 8>, ndim: u32, strides: array<u32, 8>) -> u32 {
    var physical = 0u;
    for (var d = 0u; d < ndim; d++) {
        physical += indices[d] * strides[d];
    }
    return physical;
}

// Decode flat index ke per-dim indices berdasarkan shape
fn decode(flat: u32, shape: array<u32, 8>, ndim: u32) -> array<u32, 8> {
    var indices: array<u32, 8>;
    var remaining = flat;
    // Hitung default strides dari shape
    var default_strides: array<u32, 8>;
    var s = 1u;
    for (var d = i32(ndim) - 1; d >= 0; d--) {
        default_strides[d] = s;
        s *= shape[d];
    }
    for (var d = 0u; d < ndim; d++) {
        indices[d] = remaining / default_strides[d];
        remaining  = remaining % default_strides[d];
    }
    return indices;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups)       groups:    vec3<u32>,
) {
    let wg_linear   = global_id.y * groups.x + (global_id.x / 256u);
    let local_in_wg = global_id.x % 256u;
    let out_idx     = wg_linear * 256u + local_in_wg;

    if (out_idx >= meta_out.total) { return; }

    // Decode out_idx ke indices output (tanpa axis dimension)
    let out_indices = decode(out_idx, meta_out.shape, meta_out.ndim);

    // Rebuild indices input dengan insert axis=k
    var acc = 0.0f;
    for (var k = 0u; k < ax.axis_size; k++) {
        var in_indices: array<u32, 8>;

        // Salin indices output, sisipkan k di posisi axis
        var out_d = 0u;
        for (var in_d = 0u; in_d < meta_in.ndim; in_d++) {
            if (in_d == ax.axis) {
                in_indices[in_d] = k;
            } else {
                in_indices[in_d] = out_indices[out_d];
                out_d += 1u;
            }
        }

        let phys = to_physical(in_indices, meta_in.ndim, meta_in.strides);
        acc += buf_in[phys];
    }

    buf_out[out_idx] = acc;
}
