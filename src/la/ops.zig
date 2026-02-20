const std = @import("std");
const core = @import("core.zig");
const view = @import("view.zig");
const owning = @import("owning.zig");
const cpu_scalar = @import("backend/cpu_scalar.zig");
const cpu_simd = @import("backend/cpu_simd.zig");
const blas = @import("backend/blas.zig");
const cuda = @import("backend/cuda.zig");

pub fn addInto(comptime T: type, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    return cpu_scalar.addInto(T, out, a, b);
}

pub fn subInto(comptime T: type, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    return cpu_scalar.subInto(T, out, a, b);
}

pub fn mulInto(comptime T: type, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    return cpu_scalar.mulInto(T, out, a, b);
}

pub fn divInto(comptime T: type, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    return cpu_scalar.divInto(T, out, a, b);
}

pub fn reluInplace(comptime T: type, x: view.MatMutView(T)) core.LaError!void {
    return cpu_scalar.reluInplace(T, x);
}

pub fn sum(comptime T: type, x: view.MatView(T)) core.LaError!T {
    return cpu_scalar.sumMat(T, x);
}

pub fn max(comptime T: type, x: view.MatView(T)) core.LaError!T {
    return cpu_scalar.maxMat(T, x);
}

pub fn norm2(comptime T: type, x: view.MatView(T)) core.LaError!T {
    return cpu_scalar.norm2Mat(T, x);
}

pub fn dot(comptime T: type, a: view.VecView(T), b: view.VecView(T)) core.LaError!T {
    return cpu_scalar.dot(T, a, b);
}

pub fn matmulInto(
    comptime T: type,
    out: view.MatMutView(T),
    a: view.MatView(T),
    b: view.MatView(T),
    opts: core.MatmulOpts,
) core.LaError!void {
    return switch (opts.backend) {
        .auto, .cpu_scalar => cpu_scalar.matmulInto(T, out, a, b, opts),
        .cpu_simd => cpu_simd.matmulInto(T, out, a, b, opts),
        .blas => blas.matmulInto(T, out, a, b, opts),
        .cuda => cuda.matmulInto(T, out, a, b, opts),
    };
}

pub fn cloneAlloc(comptime T: type, alloc: std.mem.Allocator, x: view.MatView(T)) core.LaError!owning.Matrix(T) {
    var out = try owning.Matrix(T).init(alloc, x.rows, x.cols);
    errdefer out.deinit(alloc);

    var i: usize = 0;
    while (i < x.rows) : (i += 1) {
        var j: usize = 0;
        while (j < x.cols) : (j += 1) {
            out.asMatMutView().set(i, j, x.get(i, j));
        }
    }
    return out;
}

pub fn addAlloc(comptime T: type, alloc: std.mem.Allocator, a: view.MatView(T), b: view.MatView(T)) core.LaError!owning.Matrix(T) {
    var out = try owning.Matrix(T).init(alloc, a.rows, a.cols);
    errdefer out.deinit(alloc);
    try addInto(T, out.asMatMutView(), a, b);
    return out;
}

pub fn subAlloc(comptime T: type, alloc: std.mem.Allocator, a: view.MatView(T), b: view.MatView(T)) core.LaError!owning.Matrix(T) {
    var out = try owning.Matrix(T).init(alloc, a.rows, a.cols);
    errdefer out.deinit(alloc);
    try subInto(T, out.asMatMutView(), a, b);
    return out;
}

pub fn mulAlloc(comptime T: type, alloc: std.mem.Allocator, a: view.MatView(T), b: view.MatView(T)) core.LaError!owning.Matrix(T) {
    var out = try owning.Matrix(T).init(alloc, a.rows, a.cols);
    errdefer out.deinit(alloc);
    try mulInto(T, out.asMatMutView(), a, b);
    return out;
}

pub fn divAlloc(comptime T: type, alloc: std.mem.Allocator, a: view.MatView(T), b: view.MatView(T)) core.LaError!owning.Matrix(T) {
    var out = try owning.Matrix(T).init(alloc, a.rows, a.cols);
    errdefer out.deinit(alloc);
    try divInto(T, out.asMatMutView(), a, b);
    return out;
}

pub fn reluAlloc(comptime T: type, alloc: std.mem.Allocator, x: view.MatView(T)) core.LaError!owning.Matrix(T) {
    var out = try cloneAlloc(T, alloc, x);
    errdefer out.deinit(alloc);
    try reluInplace(T, out.asMatMutView());
    return out;
}

pub fn matmulAlloc(
    comptime T: type,
    alloc: std.mem.Allocator,
    a: view.MatView(T),
    b: view.MatView(T),
    opts: core.MatmulOpts,
) core.LaError!owning.Matrix(T) {
    const a_rows = if (opts.trans_a) a.cols else a.rows;
    const b_cols = if (opts.trans_b) b.rows else b.cols;

    var out = try owning.Matrix(T).init(alloc, a_rows, b_cols);
    errdefer out.deinit(alloc);

    out.fill(@as(T, 0));
    try matmulInto(T, out.asMatMutView(), a, b, opts);
    return out;
}
