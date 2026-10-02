/// Jumlah elemen dari shape.
pub fn num_elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Stride untuk layout contiguous (row-major).
pub fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() { return vec![]; }
    let mut strides = vec![1; shape.len()];
    let mut current = 1;
    for i in (0..shape.len() - 1).rev() {
        current *= shape[i + 1];
        strides[i] = current;
    }
    strides
}

/// Konversi logical flat index → physical index via strides.
pub fn logical_to_physical(logical_flat: usize, shape: &[usize], strides: &[usize]) -> usize {
    let default_strides = calculate_strides(shape);
    let mut remaining = logical_flat;
    let mut physical = 0;
    for dim in 0..shape.len() {
        let idx = remaining / default_strides[dim];
        remaining %= default_strides[dim];
        physical += idx * strides[dim];
    }
    physical
}