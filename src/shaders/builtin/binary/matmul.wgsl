// Matrix multiplication: C = A @ B
// A shape: [M, K]
// B shape: [K, N]
// C shape: [M, N]
//
// Tiled matmul — tiap workgroup handle tile 16x16 dari C
// Shared memory untuk cache tile A dan B
//
// Bindings:
//   0: buf_a   (read) — matrix A, shape [M, K]
//   1: buf_b   (read) — matrix B, shape [K, N]
//   2: buf_out (write) — matrix C, shape [M, N]
//   3: dims    (uniform) — M, K, N, batch_offset_a, batch_offset_b, batch_offset_c

struct Dims {
    M:              u32,
    K:              u32,
    N:              u32,
    batch_offset_a: u32,
    batch_offset_b: u32,
    batch_offset_c: u32,
    
    stride_a0:      u32,  // strides A dim 0 (row stride)
    stride_a1:      u32,  // strides A dim 1 (col stride)
    stride_b0:      u32,  // strides B dim 0
    stride_b1:      u32,  // strides B dim 1
}

@group(0) @binding(0) var<storage, read>       buf_a:   array<f32>;
@group(0) @binding(1) var<storage, read>       buf_b:   array<f32>;
@group(0) @binding(2) var<storage, read_write> buf_out: array<f32>;
@group(0) @binding(3) var<uniform>             dims:    Dims;

const TILE: u32 = 16u;

var<workgroup> tile_a: array<f32, 256>;  // TILE x TILE = 16x16
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
    @builtin(workgroup_id)         wg_id:     vec3<u32>,
) {
    let row = global_id.y;  // baris di C (0..M)
    let col = global_id.x;  // kolom di C (0..N)

    let local_row = local_id.y;
    let local_col = local_id.x;

    var acc = 0.0f;

    // Loop tile sepanjang K
    let num_tiles = (dims.K + TILE - 1u) / TILE;

    for (var t = 0u; t < num_tiles; t++) {
        // Load tile A ke shared memory
        // Thread (local_row, local_col) load A[row, t*TILE + local_col]
        let a_col = t * TILE + local_col;
        if (row < dims.M && a_col < dims.K) {
            tile_a[local_row * TILE + local_col] =
                buf_a[dims.batch_offset_a + row * dims.stride_a0 + a_col * dims.stride_a1];
        } else {
            tile_a[local_row * TILE + local_col] = 0.0;
        }

        // Load tile B
        let b_row = t * TILE + local_row;
        if (b_row < dims.K && col < dims.N) {
            tile_b[local_row * TILE + local_col] =
                buf_b[dims.batch_offset_b + b_row * dims.stride_b0 + col * dims.stride_b1];
        } else {
            tile_b[local_row * TILE + local_col] = 0.0;
        }

        workgroupBarrier();

        // Komputasi partial dot product
        // tile_a[local_row, k] = tile_a[local_row * TILE + k]
        // tile_b[k, local_col] = tile_b[k * TILE + local_col]
        for (var k = 0u; k < TILE; k++) {
            acc += tile_a[local_row * TILE + k] * tile_b[k * TILE + local_col];
        }

        workgroupBarrier();
    }

    // Tulis hasil ke C
    if (row < dims.M && col < dims.N) {
        buf_out[dims.batch_offset_c + row * dims.N + col] = acc;
    }
}
