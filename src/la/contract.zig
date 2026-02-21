const std = @import("std");
const core = @import("core.zig");
const view = @import("view.zig");

pub fn validateMat(x: anytype) core.LaError!void {
    if (x.rows > 1 and x.row_stride == 0) return error.InvalidStride;
    if (x.cols > 1 and x.col_stride == 0) return error.InvalidStride;
}

pub fn validateVec(x: anytype) core.LaError!void {
    if (x.len > 1 and x.stride == 0) return error.InvalidStride;
}

pub fn expectSameShape(rows_a: usize, cols_a: usize, rows_b: usize, cols_b: usize) core.LaError!void {
    if (rows_a != rows_b or cols_a != cols_b) return error.DimensionMismatch;
}

pub fn expectNoOverlapMat(
    comptime T: type,
    out: view.MatMutView(T),
    a: view.MatView(T),
) core.LaError!void {
    if (try matViewsOverlap(T, out, a)) return error.AliasViolation;
}

pub fn expectNoPartialOverlapMatAllowExact(
    comptime T: type,
    out: view.MatMutView(T),
    a: view.MatView(T),
) core.LaError!void {
    if (sameMatDescriptor(T, out, a)) return;
    if (try matViewsOverlap(T, out, a)) return error.AliasViolation;
}

pub const ElementwiseAliasPolicy = enum {
    allow_any,
    allow_exact_only,
    forbid_all,
};

pub fn checkElementwiseBinaryInto(
    comptime T: type,
    out: view.MatMutView(T),
    a: view.MatView(T),
    b: view.MatView(T),
    alias: ElementwiseAliasPolicy,
) core.LaError!void {
    try validateMat(out);
    try validateMat(a);
    try validateMat(b);

    try expectSameShape(out.rows, out.cols, a.rows, a.cols);
    try expectSameShape(out.rows, out.cols, b.rows, b.cols);

    switch (alias) {
        .allow_any => {},
        .allow_exact_only => {
            try expectNoPartialOverlapMatAllowExact(T, out, a);
            try expectNoPartialOverlapMatAllowExact(T, out, b);
        },
        .forbid_all => {
            try expectNoOverlapMat(T, out, a);
            try expectNoOverlapMat(T, out, b);
        },
    }
}

pub const MatmulDims = struct {
    m: usize,
    n: usize,
    k: usize,
};

pub fn checkMatmulDims(
    comptime T: type,
    out: view.MatMutView(T),
    a: view.MatView(T),
    b: view.MatView(T),
) core.LaError!MatmulDims {
    try validateMat(out);
    try validateMat(a);
    try validateMat(b);

    if (a.cols != b.rows) return error.DimensionMismatch;
    if (out.rows != a.rows or out.cols != b.cols) return error.DimensionMismatch;

    try expectNoOverlapMat(T, out, a);
    try expectNoOverlapMat(T, out, b);

    return .{ .m = a.rows, .n = b.cols, .k = a.cols };
}

fn sameMatDescriptor(
    comptime T: type,
    out: view.MatMutView(T),
    a: view.MatView(T),
) bool {
    return @intFromPtr(out.ptr) == @intFromPtr(a.ptr) and
        out.rows == a.rows and
        out.cols == a.cols and
        out.row_stride == a.row_stride and
        out.col_stride == a.col_stride;
}

const ByteRange = struct {
    start: usize,
    end: usize,
};

fn matViewsOverlap(comptime T: type, a: anytype, b: anytype) core.LaError!bool {
    if (a.rows == 0 or a.cols == 0 or b.rows == 0 or b.cols == 0) return false;

    const ar = matByteRange(T, a) orelse return error.InvalidStride;
    const br = matByteRange(T, b) orelse return error.InvalidStride;
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
