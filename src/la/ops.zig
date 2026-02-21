const std = @import("std");
const core = @import("core.zig");
const view = @import("view.zig");
const owning = @import("owning.zig");
const cpu_scalar = @import("backend/cpu_scalar.zig");
const cpu_simd = @import("backend/cpu_simd.zig");
const cfg = @import("build_options");
const blas_provider = if (cfg.enable_blas) @import("provider/blas.zig") else struct {};

const CpuImpl = enum { scalar, simd };

fn resolve(exec: core.Exec) core.LaError!CpuImpl {
    return switch (exec.simd) {
        .scalar => .scalar,
        .auto => if (cpu_simd.is_available) .simd else .scalar,
        .simd => if (cpu_simd.is_available) .simd else error.BackendUnavailable,
    };
}

fn maybeBlasMatmul(
    comptime T: type,
    exec: core.Exec,
    out: view.MatMutView(T),
    a: view.MatView(T),
    b: view.MatView(T),
) core.LaError!bool {
    // Keep explicit scalar/simd requests deterministic; BLAS is only an auto-mode provider.
    if (exec.simd != .auto) return false;

    if (comptime cfg.enable_blas) {
        if (!blas_provider.is_available) return false;
        if (!blas_provider.canUse(T, out, a, b)) return false;

        blas_provider.matmulInto(T, out, a, b) catch |err| switch (err) {
            error.BackendUnavailable => return false,
            else => return err,
        };
        return true;
    }
    return false;
}

fn maybeBlasGemm(
    comptime T: type,
    exec: core.Exec,
    out: view.MatMutView(T),
    a: view.MatView(T),
    b: view.MatView(T),
    opts: core.GemmOpts(T),
) core.LaError!bool {
    if (exec.simd != .auto) return false;

    if (comptime cfg.enable_blas) {
        if (!blas_provider.is_available) return false;
        if (!blas_provider.canUse(T, out, a, b)) return false;

        blas_provider.gemmInto(T, out, a, b, opts) catch |err| switch (err) {
            error.BackendUnavailable => return false,
            else => return err,
        };
        return true;
    }
    return false;
}

pub fn addIntoEx(comptime T: type, exec: core.Exec, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    return switch (try resolve(exec)) {
        .scalar => cpu_scalar.addInto(T, out, a, b),
        .simd => cpu_simd.addInto(T, out, a, b),
    };
}

pub fn addInto(comptime T: type, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    return addIntoEx(T, .{}, out, a, b);
}

pub fn subIntoEx(comptime T: type, exec: core.Exec, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    return switch (try resolve(exec)) {
        .scalar => cpu_scalar.subInto(T, out, a, b),
        .simd => cpu_simd.subInto(T, out, a, b),
    };
}

pub fn subInto(comptime T: type, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    return subIntoEx(T, .{}, out, a, b);
}

pub fn mulIntoEx(comptime T: type, exec: core.Exec, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    return switch (try resolve(exec)) {
        .scalar => cpu_scalar.mulInto(T, out, a, b),
        .simd => cpu_simd.mulInto(T, out, a, b),
    };
}

pub fn mulInto(comptime T: type, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    return mulIntoEx(T, .{}, out, a, b);
}

pub fn divIntoEx(comptime T: type, exec: core.Exec, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    return switch (try resolve(exec)) {
        .scalar => cpu_scalar.divInto(T, out, a, b),
        .simd => cpu_simd.divInto(T, out, a, b),
    };
}

pub fn divInto(comptime T: type, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    return divIntoEx(T, .{}, out, a, b);
}

pub fn reluInplaceEx(comptime T: type, exec: core.Exec, x: view.MatMutView(T)) core.LaError!void {
    return switch (try resolve(exec)) {
        .scalar => cpu_scalar.reluInplace(T, x),
        .simd => cpu_simd.reluInplace(T, x),
    };
}

pub fn reluInplace(comptime T: type, x: view.MatMutView(T)) core.LaError!void {
    return reluInplaceEx(T, .{}, x);
}

pub fn sumEx(comptime T: type, exec: core.Exec, x: view.MatView(T)) core.LaError!T {
    return switch (try resolve(exec)) {
        .scalar => cpu_scalar.sumMat(T, x),
        .simd => cpu_simd.sumMat(T, x),
    };
}

pub fn sum(comptime T: type, x: view.MatView(T)) core.LaError!T {
    return sumEx(T, .{}, x);
}

pub fn maxEx(comptime T: type, exec: core.Exec, x: view.MatView(T)) core.LaError!T {
    return switch (try resolve(exec)) {
        .scalar => cpu_scalar.maxMat(T, x),
        .simd => cpu_simd.maxMat(T, x),
    };
}

pub fn max(comptime T: type, x: view.MatView(T)) core.LaError!T {
    return maxEx(T, .{}, x);
}

pub fn norm2Ex(comptime T: type, exec: core.Exec, x: view.MatView(T)) core.LaError!T {
    return switch (try resolve(exec)) {
        .scalar => cpu_scalar.norm2Mat(T, x),
        .simd => cpu_simd.norm2Mat(T, x),
    };
}

pub fn norm2(comptime T: type, x: view.MatView(T)) core.LaError!T {
    return norm2Ex(T, .{}, x);
}

pub fn dotEx(comptime T: type, exec: core.Exec, a: view.VecView(T), b: view.VecView(T)) core.LaError!T {
    return switch (try resolve(exec)) {
        .scalar => cpu_scalar.dot(T, a, b),
        .simd => cpu_simd.dot(T, a, b),
    };
}

pub fn dot(comptime T: type, a: view.VecView(T), b: view.VecView(T)) core.LaError!T {
    return dotEx(T, .{}, a, b);
}

pub fn matmulIntoEx(comptime T: type, exec: core.Exec, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    if (try maybeBlasMatmul(T, exec, out, a, b)) return;

    return switch (try resolve(exec)) {
        .scalar => cpu_scalar.matmulInto(T, out, a, b),
        .simd => cpu_simd.matmulInto(T, out, a, b),
    };
}

pub fn matmulInto(comptime T: type, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    return matmulIntoEx(T, .{}, out, a, b);
}

pub fn gemmIntoEx(
    comptime T: type,
    exec: core.Exec,
    out: view.MatMutView(T),
    a: view.MatView(T),
    b: view.MatView(T),
    opts: core.GemmOpts(T),
) core.LaError!void {
    if (try maybeBlasGemm(T, exec, out, a, b, opts)) return;

    return switch (try resolve(exec)) {
        .scalar => cpu_scalar.gemmInto(T, out, a, b, opts),
        .simd => cpu_simd.gemmInto(T, out, a, b, opts),
    };
}

pub fn gemmInto(
    comptime T: type,
    out: view.MatMutView(T),
    a: view.MatView(T),
    b: view.MatView(T),
    opts: core.GemmOpts(T),
) core.LaError!void {
    return gemmIntoEx(T, .{}, out, a, b, opts);
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

pub fn addAllocEx(comptime T: type, alloc: std.mem.Allocator, exec: core.Exec, a: view.MatView(T), b: view.MatView(T)) core.LaError!owning.Matrix(T) {
    var out = try owning.Matrix(T).init(alloc, a.rows, a.cols);
    errdefer out.deinit(alloc);
    try addIntoEx(T, exec, out.asMatMutView(), a, b);
    return out;
}

pub fn addAlloc(comptime T: type, alloc: std.mem.Allocator, a: view.MatView(T), b: view.MatView(T)) core.LaError!owning.Matrix(T) {
    return addAllocEx(T, alloc, .{}, a, b);
}

pub fn subAllocEx(comptime T: type, alloc: std.mem.Allocator, exec: core.Exec, a: view.MatView(T), b: view.MatView(T)) core.LaError!owning.Matrix(T) {
    var out = try owning.Matrix(T).init(alloc, a.rows, a.cols);
    errdefer out.deinit(alloc);
    try subIntoEx(T, exec, out.asMatMutView(), a, b);
    return out;
}

pub fn subAlloc(comptime T: type, alloc: std.mem.Allocator, a: view.MatView(T), b: view.MatView(T)) core.LaError!owning.Matrix(T) {
    return subAllocEx(T, alloc, .{}, a, b);
}

pub fn mulAllocEx(comptime T: type, alloc: std.mem.Allocator, exec: core.Exec, a: view.MatView(T), b: view.MatView(T)) core.LaError!owning.Matrix(T) {
    var out = try owning.Matrix(T).init(alloc, a.rows, a.cols);
    errdefer out.deinit(alloc);
    try mulIntoEx(T, exec, out.asMatMutView(), a, b);
    return out;
}

pub fn mulAlloc(comptime T: type, alloc: std.mem.Allocator, a: view.MatView(T), b: view.MatView(T)) core.LaError!owning.Matrix(T) {
    return mulAllocEx(T, alloc, .{}, a, b);
}

pub fn divAllocEx(comptime T: type, alloc: std.mem.Allocator, exec: core.Exec, a: view.MatView(T), b: view.MatView(T)) core.LaError!owning.Matrix(T) {
    var out = try owning.Matrix(T).init(alloc, a.rows, a.cols);
    errdefer out.deinit(alloc);
    try divIntoEx(T, exec, out.asMatMutView(), a, b);
    return out;
}

pub fn divAlloc(comptime T: type, alloc: std.mem.Allocator, a: view.MatView(T), b: view.MatView(T)) core.LaError!owning.Matrix(T) {
    return divAllocEx(T, alloc, .{}, a, b);
}

pub fn reluAllocEx(comptime T: type, alloc: std.mem.Allocator, exec: core.Exec, x: view.MatView(T)) core.LaError!owning.Matrix(T) {
    var out = try cloneAlloc(T, alloc, x);
    errdefer out.deinit(alloc);
    try reluInplaceEx(T, exec, out.asMatMutView());
    return out;
}

pub fn reluAlloc(comptime T: type, alloc: std.mem.Allocator, x: view.MatView(T)) core.LaError!owning.Matrix(T) {
    return reluAllocEx(T, alloc, .{}, x);
}

pub fn matmulAllocEx(comptime T: type, alloc: std.mem.Allocator, exec: core.Exec, a: view.MatView(T), b: view.MatView(T)) core.LaError!owning.Matrix(T) {
    var out = try owning.Matrix(T).init(alloc, a.rows, b.cols);
    errdefer out.deinit(alloc);
    try matmulIntoEx(T, exec, out.asMatMutView(), a, b);
    return out;
}

pub fn matmulAlloc(comptime T: type, alloc: std.mem.Allocator, a: view.MatView(T), b: view.MatView(T)) core.LaError!owning.Matrix(T) {
    return matmulAllocEx(T, alloc, .{}, a, b);
}
