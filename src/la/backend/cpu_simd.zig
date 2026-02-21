const core = @import("../core.zig");
const view = @import("../view.zig");
const cfg = @import("build_options");

const has_kernel_impl = false;
pub const is_available = cfg.enable_simd and has_kernel_impl;

pub fn addInto(comptime T: type, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    _ = out;
    _ = a;
    _ = b;
    return error.BackendUnavailable;
}

pub fn subInto(comptime T: type, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    _ = out;
    _ = a;
    _ = b;
    return error.BackendUnavailable;
}

pub fn mulInto(comptime T: type, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    _ = out;
    _ = a;
    _ = b;
    return error.BackendUnavailable;
}

pub fn divInto(comptime T: type, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    _ = out;
    _ = a;
    _ = b;
    return error.BackendUnavailable;
}

pub fn reluInplace(comptime T: type, x: view.MatMutView(T)) core.LaError!void {
    _ = x;
    return error.BackendUnavailable;
}

pub fn sumMat(comptime T: type, x: view.MatView(T)) core.LaError!T {
    _ = x;
    return error.BackendUnavailable;
}

pub fn maxMat(comptime T: type, x: view.MatView(T)) core.LaError!T {
    _ = x;
    return error.BackendUnavailable;
}

pub fn norm2Mat(comptime T: type, x: view.MatView(T)) core.LaError!T {
    _ = x;
    return error.BackendUnavailable;
}

pub fn dot(comptime T: type, a: view.VecView(T), b: view.VecView(T)) core.LaError!T {
    _ = a;
    _ = b;
    return error.BackendUnavailable;
}

pub fn matmulInto(comptime T: type, out: view.MatMutView(T), a: view.MatView(T), b: view.MatView(T)) core.LaError!void {
    _ = out;
    _ = a;
    _ = b;
    return error.BackendUnavailable;
}

pub fn gemmInto(
    comptime T: type,
    out: view.MatMutView(T),
    a: view.MatView(T),
    b: view.MatView(T),
    opts: core.GemmOpts(T),
) core.LaError!void {
    _ = out;
    _ = a;
    _ = b;
    _ = opts;
    return error.BackendUnavailable;
}
