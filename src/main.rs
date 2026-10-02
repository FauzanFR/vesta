use vesta::{Device, Tensor};

fn assert_close(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 1e-4, "{name}[{i}]: got {g}, want {w}");
    }
    println!("{name}: ok {got:?}");
}

fn main() {
    let device = Device::new().expect("no usable GPU adapter");

    // A (2x3) @ B (3x2)
    let a = Tensor::from_vec(&device, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Tensor::from_vec(&device, vec![3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_close(
        "matmul 2x3 @ 3x2",
        &a.matmul(&b).to_cpu().unwrap(),
        &[22.0, 28.0, 49.0, 64.0],
    );

    // Against an all-ones matrix, each output row is a row sum of A.
    let ones = Tensor::from_vec(&device, vec![3, 4], &[1.0; 12]);
    assert_close(
        "matmul 2x3 @ ones(3x4)",
        &a.matmul(&ones).to_cpu().unwrap(),
        &[6.0, 6.0, 6.0, 6.0, 15.0, 15.0, 15.0, 15.0],
    );

    // Lazy transpose (zero-copy strided view): A^T @ A -> 3x3
    let at = a.T();
    assert_close(
        "A^T @ A (lazy transpose)",
        &at.matmul(&a).to_cpu().unwrap(),
        &[17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0],
    );
}
