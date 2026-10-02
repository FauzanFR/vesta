pub fn get_shader(name: &str) -> &'static str {
    match name {
        // init
        "fill"    => include_str!("builtin/init/fill.wgsl"),
        "random"  => include_str!("builtin/init/random.wgsl"),
        "arange"  => include_str!("builtin/init/arange.wgsl"),
        // binary
        "add"      => include_str!("builtin/binary/add.wgsl"),
        "subtract" => include_str!("builtin/binary/subtract.wgsl"),
        "multiply" => include_str!("builtin/binary/multiply.wgsl"),
        "divide"   => include_str!("builtin/binary/divide.wgsl"),
        "max"      => include_str!("builtin/binary/max.wgsl"),
        "min"      => include_str!("builtin/binary/min.wgsl"),
        "pow"      => include_str!("builtin/binary/pow.wgsl"),
        "matmul"   => include_str!("builtin/binary/matmul.wgsl"),
        // unary
        "relu"        => include_str!("builtin/unary/relu.wgsl"),
        "sigmoid"     => include_str!("builtin/unary/sigmoid.wgsl"),
        "tanh"        => include_str!("builtin/unary/tanh.wgsl"),
        "exp"         => include_str!("builtin/unary/exp.wgsl"),
        "log"         => include_str!("builtin/unary/log.wgsl"),
        "sqrt"        => include_str!("builtin/unary/sqrt.wgsl"),
        "abs"         => include_str!("builtin/unary/abs.wgsl"),
        "neg"         => include_str!("builtin/unary/neg.wgsl"),
        "softplus"    => include_str!("builtin/unary/softplus.wgsl"),
        "clip"        => include_str!("builtin/unary/clip.wgsl"),
        "leaky_relu"  => include_str!("builtin/unary/leaky_relu.wgsl"),
        "elu"         => include_str!("builtin/unary/elu.wgsl"),
        "reduce_sum"  => include_str!("builtin/unary/reduce_sum.wgsl"),
        "reduce_max"  => include_str!("builtin/unary/reduce_max.wgsl"),
        "reduce_axis" => include_str!("builtin/unary/reduce_axis.wgsl"),
        _ => panic!("shader tidak ditemukan: {}", name),
    }
}