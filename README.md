# Vesta

A small GPU tensor engine written in Rust on top of [wgpu](https://wgpu.rs), with hand-written WGSL compute kernels.

> **Status: early prototype.** The core pieces work and are checked on small tensors against hand-computed values, but several large-tensor (multi-buffer) paths are incomplete. See [Known limitations](#known-limitations) — they are listed explicitly rather than hidden.

## What works

- **Creation:** `from_vec`, `zeros`, `ones`, `fill`, `arange`, `random`
- **Elementwise (binary, same shape):** `add`, `subtract`, `multiply`, `divide`, `max`, `min`, `pow`
- **Elementwise (unary):** `relu`, `leaky_relu`, `elu`, `sigmoid`, `tanh`, `softplus`, `exp`, `log`, `sqrt`, `abs`, `neg`, `clip`
- **Reductions:** `sum`, `mean`, `sum_axis`, `mean_axis`
- **Linear algebra:** `matmul` (2-D × 2-D, 16×16 tiled shared-memory kernel)
- **Views:** `T()` — zero-copy lazy transpose via strides
- **Readback and bookkeeping:** `to_cpu`, `Device::vram_used`, `Device::tensor_count`

All tensors are `f32`. About 25 WGSL kernels live in `src/shaders/builtin/`.

## Quick start

Requires Rust 1.85+ (edition 2024) and a GPU adapter supported by wgpu.

```bash
cargo run
```

`src/main.rs` runs a few matmuls (including one on a lazily transposed view) and asserts the results.

```rust
use vesta::{Device, Tensor};

let device = Device::new()?;
let a = Tensor::arange(&device, vec![4, 3], 0.0, 1.0); // [[0,1,2],[3,4,5],[6,7,8],[9,10,11]]
let b = Tensor::ones(&device, vec![3, 2]);
let c = a.matmul(&b);                                   // [4, 2]
println!("{:?}", c.to_cpu()?);                          // [3, 3, 12, 12, 21, 21, 30, 30]
println!("sum = {}", c.sum());                          // 132
```

### Choosing a backend

The wgpu backend is picked automatically. To force one, set `WGPU_BACKEND` to `dx12`, `vulkan`, `metal` or `gl`:

```powershell
$env:WGPU_BACKEND = "dx12"; cargo run      # PowerShell
```
```bash
WGPU_BACKEND=vulkan cargo run              # bash
```

The author hit problems with the Vulkan backend on an integrated GPU and uses DX12 there.

## How it works

- **Chunked storage.** A tensor's data is a list of `wgpu::Buffer` chunks, each sized to the device's `max_buffer_size` / `max_storage_buffer_binding_size`. Ops loop over chunks and dispatch once per chunk.
- **Strided views.** A `ViewMeta` block (shape and strides up to 8 dims, chunk offset) is passed to each kernel, so a transposed view needs no copy.
- **Dispatch and caching.** Shader modules, bind-group layouts and compute pipelines are cached (pipelines keyed by shader name and a hash of the binding layout). 1-D dispatches fold into a 2-D grid to get past the 65,535-workgroup limit.
- **Reductions.** `sum` is a multi-pass tree reduction in workgroup shared memory with ping-pong buffers; per-chunk partial sums are added on the CPU.

## Known limitations

Checked by running small tensors with the device's buffer limit artificially lowered so they split into two chunks:

- **Chunking only starts at the default wgpu limits.** The device is created with default limits (128 MiB storage binding), so tensors above ~33.5 M `f32` elements are chunked even if the GPU could hold more.
- **Multi-chunk support is partial.**
  - Works: elementwise ops on contiguous tensors, `sum`/`mean`, and `sum_axis` (via a correct but slow CPU fallback).
  - Panics instead of returning wrong results: elementwise ops on non-contiguous views (e.g. `a.T()`) when the tensor has more than one chunk, and `matmul` whose output needs more than one chunk.
  - `matmul` also requires both inputs to be single-chunk.
  - The `reduce_axis` kernel is not chunk-aware yet, so its GPU path is only used for single-chunk tensors.
- **`matmul` is 2-D × 2-D only**, with no batch dimensions.
- **`f32` only**, with `u32` indexing (at most 2³² elements per tensor).
- **Performance is not measured yet.** Every op allocates small metadata/uniform buffers, builds its own bind group and submits on its own; there is no command batching or buffer pooling. There are no benchmarks.
- **No automated test suite and no autograd yet.**
- `reduce_max` has a kernel but no Rust API.

## Roadmap

1. Request the adapter's real limits and make the chunk size configurable, so multi-chunk paths can be tested with tiny tensors.
2. Make the elementwise and `reduce_axis` kernels correct for strided multi-chunk tensors; support multi-chunk `matmul`.
3. Add a test suite against a CPU reference.
4. Benchmark against a CPU baseline, then batch commands and pool buffers.
5. Batched matmul and more dtypes.

## License

Apache-2.0 — see [LICENSE](LICENSE).
