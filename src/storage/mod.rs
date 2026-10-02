mod meta;
mod slice;
mod alloc;

pub use meta::ViewMeta;
pub use slice::TensorSlice;
pub use alloc::{alloc_slice, upload_slice};