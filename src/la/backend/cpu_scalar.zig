const core = @import("../core.zig");
const view = @import("../view.zig");
const contract = @import("../contract.zig");

pub fn addInto(comptime T: type, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    core.assertNumericType(T);
    try contract.checkElementwiseBinaryInto(T, out, a, b, .allow_exact_only);

    var i: usize = 0;
    while (i < out.rows) : (i += 1) {
        var j: usize = 0;
        while (j < out.cols) : (j += 1) {
            out.set(i, j, a.get(i, j) + b.get(i, j));
        }
    }
}

pub fn subInto(comptime T: type, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    core.assertNumericType(T);
    try contract.checkElementwiseBinaryInto(T, out, a, b, .allow_exact_only);

    var i: usize = 0;
    while (i < out.rows) : (i += 1) {
        var j: usize = 0;
        while (j < out.cols) : (j += 1) {
            out.set(i, j, a.get(i, j) - b.get(i, j));
        }
    }
}

pub fn mulInto(comptime T: type, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    core.assertNumericType(T);
    try contract.checkElementwiseBinaryInto(T, out, a, b, .allow_exact_only);

    var i: usize = 0;
    while (i < out.rows) : (i += 1) {
        var j: usize = 0;
        while (j < out.cols) : (j += 1) {
            out.set(i, j, a.get(i, j) * b.get(i, j));
        }
    }
}

pub fn divInto(comptime T: type, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    core.assertNumericType(T);
    try contract.checkElementwiseBinaryInto(T, out, a, b, .allow_exact_only);

    var i: usize = 0;
    while (i < out.rows) : (i += 1) {
        var j: usize = 0;
        while (j < out.cols) : (j += 1) {
            out.set(i, j, divValue(T, a.get(i, j), b.get(i, j)));
        }
    }
}

pub fn reluInplace(comptime T: type, x: view.MatMutView(T)) core.LaError!void {
    core.assertNumericType(T);
    try contract.validateMat(x);

    var i: usize = 0;
    while (i < x.rows) : (i += 1) {
        var j: usize = 0;
        while (j < x.cols) : (j += 1) {
            x.set(i, j, reluValue(T, x.get(i, j)));
        }
    }
}

pub fn sumMat(comptime T: type, x: view.MatView(T)) core.LaError!T {
    core.assertNumericType(T);
    try contract.validateMat(x);

    var acc: T = @as(T, 0);
    var i: usize = 0;
    while (i < x.rows) : (i += 1) {
        var j: usize = 0;
        while (j < x.cols) : (j += 1) {
            acc += x.get(i, j);
        }
    }
    return acc;
}

pub fn maxMat(comptime T: type, x: view.MatView(T)) core.LaError!T {
    core.assertNumericType(T);
    try contract.validateMat(x);
    if (x.rows == 0 or x.cols == 0) return error.DimensionMismatch;

    var best = x.get(0, 0);
    var i: usize = 0;
    while (i < x.rows) : (i += 1) {
        var j: usize = 0;
        while (j < x.cols) : (j += 1) {
            const value = x.get(i, j);
            if (value > best) best = value;
        }
    }
    return best;
}

pub fn norm2Mat(comptime T: type, x: view.MatView(T)) core.LaError!T {
    core.assertNumericType(T);
    if (!core.isFloatType(T)) return error.UnsupportedType;
    try contract.validateMat(x);

    var acc: T = @as(T, 0);
    var i: usize = 0;
    while (i < x.rows) : (i += 1) {
        var j: usize = 0;
        while (j < x.cols) : (j += 1) {
            const value = x.get(i, j);
            acc += value * value;
        }
    }
    return @sqrt(acc);
}

pub fn dot(comptime T: type, a: view.VecView(T), b: view.VecView(T)) core.LaError!T {
    core.assertNumericType(T);
    try contract.validateVec(a);
    try contract.validateVec(b);
    if (a.len != b.len) return error.DimensionMismatch;

    var acc: T = @as(T, 0);
    var i: usize = 0;
    while (i < a.len) : (i += 1) {
        acc += a.at(i) * b.at(i);
    }
    return acc;
}

pub fn matmulInto(
    comptime T: type,
    out: view.MatMutView(T),
    a: view.MatView(T),
    b: view.MatView(T),
) core.LaError!void {
    core.assertNumericType(T);
    const dims = try contract.checkMatmulDims(T, out, a, b);

    var i: usize = 0;
    while (i < dims.m) : (i += 1) {
        var j: usize = 0;
        while (j < dims.n) : (j += 1) {
            var acc: T = @as(T, 0);
            var k: usize = 0;
            while (k < dims.k) : (k += 1) {
                acc += a.get(i, k) * b.get(k, j);
            }
            out.set(i, j, acc);
        }
    }
}

pub fn gemmInto(
    comptime T: type,
    out: view.MatMutView(T),
    a: view.MatView(T),
    b: view.MatView(T),
    opts: core.GemmOpts(T),
) core.LaError!void {
    core.assertNumericType(T);
    if (!core.isFloatType(T)) {
        if (opts.alpha != @as(T, 1) or opts.beta != @as(T, 0)) {
            return error.UnsupportedType;
        }
    }

    const dims = try contract.checkMatmulDims(T, out, a, b);

    var i: usize = 0;
    while (i < dims.m) : (i += 1) {
        var j: usize = 0;
        while (j < dims.n) : (j += 1) {
            var acc: T = @as(T, 0);
            var k: usize = 0;
            while (k < dims.k) : (k += 1) {
                acc += a.get(i, k) * b.get(k, j);
            }
            out.set(i, j, opts.alpha * acc + opts.beta * out.get(i, j));
        }
    }
}

fn divValue(comptime T: type, lhs: T, rhs: T) T {
    return switch (@typeInfo(T)) {
        .float, .comptime_float => lhs / rhs,
        .int, .comptime_int => @divTrunc(lhs, rhs),
        else => unreachable,
    };
}

fn reluValue(comptime T: type, x: T) T {
    if (core.isFloatType(T) or core.isSignedIntType(T)) {
        const zero = @as(T, 0);
        return if (x > zero) x else zero;
    }
    return x;
}
