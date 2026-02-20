const std = @import("std");
const core = @import("../core.zig");
const view = @import("../view.zig");

pub fn addInto(comptime T: type, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    core.assertNumericType(T);
    try validateMat(out);
    try validateMat(a);
    try validateMat(b);
    try expectSameShape(out.rows, out.cols, a.rows, a.cols);
    try expectSameShape(out.rows, out.cols, b.rows, b.cols);

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
    try validateMat(out);
    try validateMat(a);
    try validateMat(b);
    try expectSameShape(out.rows, out.cols, a.rows, a.cols);
    try expectSameShape(out.rows, out.cols, b.rows, b.cols);

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
    try validateMat(out);
    try validateMat(a);
    try validateMat(b);
    try expectSameShape(out.rows, out.cols, a.rows, a.cols);
    try expectSameShape(out.rows, out.cols, b.rows, b.cols);

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
    try validateMat(out);
    try validateMat(a);
    try validateMat(b);
    try expectSameShape(out.rows, out.cols, a.rows, a.cols);
    try expectSameShape(out.rows, out.cols, b.rows, b.cols);

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
    try validateMat(x);

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
    try validateMat(x);

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
    try validateMat(x);
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
    try validateMat(x);

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
    try validateVec(a);
    try validateVec(b);
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
    opts: core.MatmulOpts,
) core.LaError!void {
    core.assertNumericType(T);
    try validateMat(out);
    try validateMat(a);
    try validateMat(b);

    const a_rows = if (opts.trans_a) a.cols else a.rows;
    const a_cols = if (opts.trans_a) a.rows else a.cols;
    const b_rows = if (opts.trans_b) b.cols else b.rows;
    const b_cols = if (opts.trans_b) b.rows else b.cols;

    if (a_cols != b_rows) return error.DimensionMismatch;
    if (out.rows != a_rows or out.cols != b_cols) return error.DimensionMismatch;

    if (matViewsOverlap(T, out, a) or matViewsOverlap(T, out, b)) {
        return error.AliasViolation;
    }

    const alpha: T = if (core.isFloatType(T)) @as(T, @floatCast(opts.alpha)) else blk: {
        if (opts.alpha != 1.0) return error.UnsupportedType;
        break :blk @as(T, 1);
    };
    const beta: T = if (core.isFloatType(T)) @as(T, @floatCast(opts.beta)) else blk: {
        if (opts.beta != 0.0) return error.UnsupportedType;
        break :blk @as(T, 0);
    };

    var i: usize = 0;
    while (i < out.rows) : (i += 1) {
        var j: usize = 0;
        while (j < out.cols) : (j += 1) {
            var acc: T = @as(T, 0);
            var k: usize = 0;
            while (k < a_cols) : (k += 1) {
                const lhs = if (opts.trans_a) a.get(k, i) else a.get(i, k);
                const rhs = if (opts.trans_b) b.get(j, k) else b.get(k, j);
                acc += lhs * rhs;
            }
            out.set(i, j, alpha * acc + beta * out.get(i, j));
        }
    }
}

fn expectSameShape(rows_a: usize, cols_a: usize, rows_b: usize, cols_b: usize) core.LaError!void {
    if (rows_a != rows_b or cols_a != cols_b) return error.DimensionMismatch;
}

fn validateMat(x: anytype) core.LaError!void {
    if (x.rows > 1 and x.row_stride == 0) return error.InvalidStride;
    if (x.cols > 1 and x.col_stride == 0) return error.InvalidStride;
}

fn validateVec(x: anytype) core.LaError!void {
    if (x.len > 1 and x.stride == 0) return error.InvalidStride;
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

const ByteRange = struct {
    start: usize,
    end: usize,
};

fn matViewsOverlap(comptime T: type, a: anytype, b: anytype) bool {
    const ar = matByteRange(T, a) orelse return false;
    const br = matByteRange(T, b) orelse return false;
    return ar.start < br.end and br.start < ar.end;
}

fn matByteRange(comptime T: type, x: anytype) ?ByteRange {
    if (x.rows == 0 or x.cols == 0) return null;

    const row_span = std.math.mul(usize, x.rows - 1, x.row_stride) catch return null;
    const col_span = std.math.mul(usize, x.cols - 1, x.col_stride) catch return null;
    const max_index = std.math.add(usize, row_span, col_span) catch return null;
    const count = std.math.add(usize, max_index, 1) catch return null;
    const byte_span = std.math.mul(usize, count, @sizeOf(T)) catch return null;

    const start = @intFromPtr(x.ptr);
    const end = std.math.add(usize, start, byte_span) catch return null;
    return .{ .start = start, .end = end };
}
