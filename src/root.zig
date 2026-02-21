const std = @import("std");

const la = @import("la.zig");
pub const cpu = la.cpu;

pub const LaError = la.LaError;
pub const SimdMode = la.SimdMode;
pub const Threading = la.Threading;
pub const Exec = la.Exec;
pub const GemmOpts = la.GemmOpts;
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

fn expectBackendUnavailableFailure(outcome: anytype, op: la.OpKind) !void {
    switch (outcome) {
        .ok => return error.TestUnexpectedResult,
        .fail => |fail| {
            try std.testing.expectEqual(op, fail.op);
            try std.testing.expectEqual(la.SimdMode.simd, fail.exec.simd);
            try std.testing.expectEqual(error.BackendUnavailable, fail.err);
        },
    }
}

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
            try std.testing.expectEqual(la.SimdMode.auto, fail.exec.simd);
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
            try std.testing.expectEqual(la.SimdMode.simd, fail.exec.simd);
            try std.testing.expectEqual(error.BackendUnavailable, fail.err);
        },
    }
}

test "simd mode requires simd backend across cpu and frame compute ops" {
    var c = la.cpu(std.testing.allocator);
    defer c.deinit();
    c.simd = .simd;

    var a = try c.mat(f32, 1, 1);
    defer a.deinit(c.alloc);
    var b = try c.mat(f32, 1, 1);
    defer b.deinit(c.alloc);
    var out = try c.mat(f32, 1, 1);
    defer out.deinit(c.alloc);

    a.asMatMutView().set(0, 0, 2);
    b.asMatMutView().set(0, 0, 3);
    out.asMatMutView().set(0, 0, 0);

    var v1 = try c.vec(f32, 1);
    defer v1.deinit(c.alloc);
    var v2 = try c.vec(f32, 1);
    defer v2.deinit(c.alloc);
    v1.asVecMutView().set(0, 4);
    v2.asVecMutView().set(0, 5);

    try std.testing.expectError(error.BackendUnavailable, c.add(a, b));
    try std.testing.expectError(error.BackendUnavailable, c.sub(a, b));
    try std.testing.expectError(error.BackendUnavailable, c.mul(a, b));
    try std.testing.expectError(error.BackendUnavailable, c.div(a, b));
    try std.testing.expectError(error.BackendUnavailable, c.relu(a));
    try std.testing.expectError(error.BackendUnavailable, c.matmul(a, b));

    try std.testing.expectError(error.BackendUnavailable, c.add_into(out, a, b));
    try std.testing.expectError(error.BackendUnavailable, c.sub_into(out, a, b));
    try std.testing.expectError(error.BackendUnavailable, c.mul_into(out, a, b));
    try std.testing.expectError(error.BackendUnavailable, c.div_into(out, a, b));
    try std.testing.expectError(error.BackendUnavailable, c.relu_inplace(out));
    try std.testing.expectError(error.BackendUnavailable, c.matmul_into(out, a, b));

    try std.testing.expectError(error.BackendUnavailable, c.sum(a));
    try std.testing.expectError(error.BackendUnavailable, c.max(a));
    try std.testing.expectError(error.BackendUnavailable, c.norm2(a));
    try std.testing.expectError(error.BackendUnavailable, c.dot(v1, v2));

    try expectBackendUnavailableFailure(c.add_with_failure(a, b), .add);
    try expectBackendUnavailableFailure(c.add_into_with_failure(out, a, b), .add);
    try expectBackendUnavailableFailure(c.sum_with_failure(a), .sum);
    try expectBackendUnavailableFailure(c.dot_with_failure(v1, v2), .dot);
    try expectBackendUnavailableFailure(c.matmul_with_failure(a, b), .matmul);

    var frame = c.frame();
    defer frame.deinit();

    try std.testing.expectError(error.BackendUnavailable, frame.add(a, b));
    try std.testing.expectError(error.BackendUnavailable, frame.sub(a, b));
    try std.testing.expectError(error.BackendUnavailable, frame.mul(a, b));
    try std.testing.expectError(error.BackendUnavailable, frame.div(a, b));
    try std.testing.expectError(error.BackendUnavailable, frame.relu(a));
    try std.testing.expectError(error.BackendUnavailable, frame.matmul(a, b));

    try expectBackendUnavailableFailure(frame.add_with_failure(a, b), .add);
    try expectBackendUnavailableFailure(frame.relu_with_failure(a), .relu);
    try expectBackendUnavailableFailure(frame.matmul_with_failure(a, b), .matmul);
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

test "elementwise into alias contract allows exact alias and rejects partial overlap" {
    var c = la.cpu(std.testing.allocator);
    defer c.deinit();

    var out = try c.mat(f32, 1, 3);
    defer out.deinit(c.alloc);
    var b = try c.mat(f32, 1, 3);
    defer b.deinit(c.alloc);

    var outv = out.asMatMutView();
    outv.set(0, 0, 1);
    outv.set(0, 1, 2);
    outv.set(0, 2, 3);

    var bv = b.asMatMutView();
    bv.set(0, 0, 10);
    bv.set(0, 1, 20);
    bv.set(0, 2, 30);

    try c.add_into(out, out, b);
    const exact_alias = out.asMatView();
    try std.testing.expectApproxEqAbs(@as(f32, 11), exact_alias.get(0, 0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 22), exact_alias.get(0, 1), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 33), exact_alias.get(0, 2), 0.0001);

    const shared = try c.alloc.alloc(f32, 4);
    defer c.alloc.free(shared);
    shared[0] = 1;
    shared[1] = 2;
    shared[2] = 3;
    shared[3] = 4;

    const out_partial: la.MatMutView(f32) = .{
        .ptr = shared[0..].ptr,
        .rows = 1,
        .cols = 3,
        .row_stride = 3,
        .col_stride = 1,
    };
    const a_partial: la.MatView(f32) = .{
        .ptr = shared[1..].ptr,
        .rows = 1,
        .cols = 3,
        .row_stride = 3,
        .col_stride = 1,
    };

    try std.testing.expectError(error.AliasViolation, c.add_into(out_partial, a_partial, b));
}

test "alias overlap check fails closed on overflow" {
    var c = la.cpu(std.testing.allocator);
    defer c.deinit();

    var b = try c.mat(f32, 2, 1);
    defer b.deinit(c.alloc);
    var bv = b.asMatMutView();
    bv.set(0, 0, 1);
    bv.set(1, 0, 2);

    const shared = try c.alloc.alloc(f32, 2);
    defer c.alloc.free(shared);
    shared[0] = 3;
    shared[1] = 4;

    const huge = std.math.maxInt(usize);
    const out_over: la.MatMutView(f32) = .{
        .ptr = shared.ptr,
        .rows = 2,
        .cols = 1,
        .row_stride = huge,
        .col_stride = 1,
    };
    const a_over: la.MatView(f32) = .{
        .ptr = shared.ptr,
        .rows = 2,
        .cols = 1,
        .row_stride = huge,
        .col_stride = 1,
    };

    try std.testing.expectError(error.InvalidStride, c.add_into(out_over, a_over, b));
}

test "mat view transpose helpers" {
    var c = la.cpu(std.testing.allocator);
    defer c.deinit();

    var a = try c.mat(f32, 2, 3);
    defer a.deinit(c.alloc);

    var av = a.asMatMutView();
    av.set(0, 0, 1);
    av.set(0, 1, 2);
    av.set(0, 2, 3);
    av.set(1, 0, 4);
    av.set(1, 1, 5);
    av.set(1, 2, 6);

    const at = a.asMatView().t();
    try std.testing.expectEqual(@as(usize, 3), at.rows);
    try std.testing.expectEqual(@as(usize, 2), at.cols);
    try std.testing.expectApproxEqAbs(@as(f32, 2), at.get(1, 0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 6), at.get(2, 1), 0.0001);

    const att = at.t();
    const base = a.asMatView();
    try std.testing.expectEqual(base.rows, att.rows);
    try std.testing.expectEqual(base.cols, att.cols);
    try std.testing.expectEqual(base.row_stride, att.row_stride);
    try std.testing.expectEqual(base.col_stride, att.col_stride);

    var amt = a.asMatMutView().t();
    amt.set(2, 1, 99);
    try std.testing.expectApproxEqAbs(@as(f32, 99), a.asMatView().get(1, 2), 0.0001);
}

test "gemm_into applies alpha and beta" {
    var c = la.cpu(std.testing.allocator);
    defer c.deinit();

    var a = try c.mat(f32, 1, 2);
    defer a.deinit(c.alloc);
    var b = try c.mat(f32, 2, 1);
    defer b.deinit(c.alloc);
    var out = try c.mat(f32, 1, 1);
    defer out.deinit(c.alloc);

    var av = a.asMatMutView();
    av.set(0, 0, 1);
    av.set(0, 1, 2);

    var bv = b.asMatMutView();
    bv.set(0, 0, 3);
    bv.set(1, 0, 4);

    out.asMatMutView().set(0, 0, 10);

    try c.gemm_into(f32, out, a, b, .{ .alpha = 2, .beta = 0.5 });
    try std.testing.expectApproxEqAbs(@as(f32, 27), out.asMatView().get(0, 0), 0.0001);
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
