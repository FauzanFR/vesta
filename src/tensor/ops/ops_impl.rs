use std::ops::{Add, Sub, Mul, Div};

use super::Tensor;

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $op:ident) => {
        impl $trait<&Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor { self.$op(rhs) }
        }
        impl $trait<Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor { self.$op(&rhs) }
        }
        impl $trait<&Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor { (&self).$op(rhs) }
        }
        impl $trait<Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor { (&self).$op(&rhs) }
        }
    };
}

impl_binary_op!(Add, add, add);
impl_binary_op!(Sub, sub, subtract);
impl_binary_op!(Mul, mul, multiply);
impl_binary_op!(Div, div, divide);