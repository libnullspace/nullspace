const std = @import("std");

pub const la = @import("la.zig");
pub const cpu = la.cpu;

pub const LaError = la.LaError;
pub const SimdMode = la.SimdMode;
pub const Threading = la.Threading;
pub const Backend = la.Backend;
pub const MatmulOpts = la.MatmulOpts;
pub const OpKind = la.OpKind;
pub const Failure = la.Failure;
pub const Outcome = la.Outcome;
pub const ScalarOf = la.ScalarOf;

pub const Matrix = la.Matrix;
pub const Vector = la.Vector;
pub const TempMat = la.TempMat;
pub const TempVec = la.TempVec;

pub const VecView = la.VecView;
pub const VecMutView = la.VecMutView;
pub const MatView = la.MatView;
pub const MatMutView = la.MatMutView;

test "cpu matrix ops and frame lifecycle" {
    var c = la.cpu(std.testing.allocator);
    defer c.deinit();

    var a = try c.mat(f32, 2, 2);
    defer a.deinit(c.alloc);
    var b = try c.mat(f32, 2, 2);
    defer b.deinit(c.alloc);

    var av = a.asMatMutView();
    av.set(0, 0, 1);
    av.set(0, 1, 2);
    av.set(1, 0, 3);
    av.set(1, 1, 4);

    var bv = b.asMatMutView();
    bv.set(0, 0, 5);
    bv.set(0, 1, 6);
    bv.set(1, 0, 7);
    bv.set(1, 1, 8);

    var add_out = try c.add(a, b);
    defer add_out.deinit(c.alloc);

    const addv = add_out.asMatView();
    try std.testing.expectApproxEqAbs(@as(f32, 6), addv.get(0, 0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 12), addv.get(1, 1), 0.0001);

    var mm = try c.matmul(a, b);
    defer mm.deinit(c.alloc);

    const mmv = mm.asMatView();
    try std.testing.expectApproxEqAbs(@as(f32, 19), mmv.get(0, 0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 22), mmv.get(0, 1), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 43), mmv.get(1, 0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 50), mmv.get(1, 1), 0.0001);

    var frame = c.frame();
    defer frame.deinit();

    const tmp = try frame.relu(add_out);
    const tmpv = tmp.asMatView();
    try std.testing.expectApproxEqAbs(@as(f32, 6), tmpv.get(0, 0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 12), tmpv.get(1, 1), 0.0001);

    var cloned = try c.clone(tmp);
    defer cloned.deinit(c.alloc);
    try std.testing.expectApproxEqAbs(@as(f32, 36), try c.sum(cloned), 0.0001);
}

test "dimension mismatch reports deterministic error" {
    var c = la.cpu(std.testing.allocator);
    defer c.deinit();

    var a = try c.mat(f32, 2, 3);
    defer a.deinit(c.alloc);
    var b = try c.mat(f32, 4, 2);
    defer b.deinit(c.alloc);

    try std.testing.expectError(error.DimensionMismatch, c.matmul(a, b));
}

test "with_failure returns contextual error details" {
    var c = la.cpu(std.testing.allocator);
    defer c.deinit();

    var a = try c.mat(f32, 2, 3);
    defer a.deinit(c.alloc);
    var b = try c.mat(f32, 4, 2);
    defer b.deinit(c.alloc);

    const mm = c.matmul_with_failure(a, b);
    switch (mm) {
        .ok => return error.TestUnexpectedResult,
        .fail => |fail| {
            try std.testing.expectEqual(la.OpKind.matmul, fail.op);
            try std.testing.expectEqual(la.Backend.auto, fail.backend);
            try std.testing.expectEqual(error.DimensionMismatch, fail.err);
        },
    }

    c.simd = .simd;
    var out = try c.mat(f32, 1, 1);
    defer out.deinit(c.alloc);

    const add_res = c.add_into_with_failure(out, out, out);
    switch (add_res) {
        .ok => return error.TestUnexpectedResult,
        .fail => |fail| {
            try std.testing.expectEqual(la.OpKind.add, fail.op);
            try std.testing.expectEqual(la.Backend.cpu_simd, fail.backend);
            try std.testing.expectEqual(error.BackendUnavailable, fail.err);
        },
    }
}

test "plain error and with_failure error stay aligned" {
    var c = la.cpu(std.testing.allocator);
    defer c.deinit();

    var a = try c.mat(f32, 2, 3);
    defer a.deinit(c.alloc);
    var b = try c.mat(f32, 4, 2);
    defer b.deinit(c.alloc);

    try std.testing.expectError(error.DimensionMismatch, c.matmul(a, b));
    const with_fail = c.matmul_with_failure(a, b);
    switch (with_fail) {
        .ok => return error.TestUnexpectedResult,
        .fail => |fail| try std.testing.expectEqual(error.DimensionMismatch, fail.err),
    }
}

test "dot and norm2" {
    var c = la.cpu(std.testing.allocator);
    defer c.deinit();

    var v1 = try c.vec(f32, 3);
    defer v1.deinit(c.alloc);
    var v2 = try c.vec(f32, 3);
    defer v2.deinit(c.alloc);

    var v1m = v1.asVecMutView();
    v1m.set(0, 1);
    v1m.set(1, 2);
    v1m.set(2, 3);

    var v2m = v2.asVecMutView();
    v2m.set(0, 4);
    v2m.set(1, 5);
    v2m.set(2, 6);

    const dot = try c.dot(v1, v2);
    try std.testing.expectApproxEqAbs(@as(f32, 32), dot, 0.0001);

    var m = try c.mat(f32, 1, 3);
    defer m.deinit(c.alloc);
    var mv = m.asMatMutView();
    mv.set(0, 0, 3);
    mv.set(0, 1, 4);
    mv.set(0, 2, 0);

    const norm = try c.norm2(m);
    try std.testing.expectApproxEqAbs(@as(f32, 5), norm, 0.0001);
}

test "owned slice transfer and frame reset contract" {
    const alloc = std.testing.allocator;

    var raw = try alloc.alloc(f32, 4);
    raw[0] = 1;
    raw[1] = 2;
    raw[2] = 3;
    raw[3] = 4;

    var m = try la.Matrix(f32).fromOwnedSliceAssumeAllocator(raw, 2, 2);
    const owned_back = m.toOwnedSlice();
    try std.testing.expectEqual(@as(usize, 0), m.rows);
    try std.testing.expectEqual(@as(usize, 0), m.cols);
    alloc.free(owned_back);

    var c = la.cpu(alloc);
    defer c.deinit();

    var frame = c.frame();
    defer frame.deinit();
    try frame.reset();
}

test "temp wrappers are frame-constructed only" {
    try std.testing.expect(@typeInfo(la.TempMat(f32)) == .@"opaque");
    try std.testing.expect(@typeInfo(la.TempVec(f32)) == .@"opaque");
    try std.testing.expect(!@hasDecl(la.TempMat(f32), "init"));
    try std.testing.expect(!@hasDecl(la.TempVec(f32), "init"));
}
