const std = @import("std");

pub const LaError = error{
    OutOfMemory,
    DimensionMismatch,
    InvalidStride,
    NonContiguousRequired,
    AliasViolation,
    BackendUnavailable,
    UnsupportedType,
};

pub const SimdMode = enum { auto, scalar, simd };
pub const Threading = enum { single, multi };
pub const Backend = enum { auto, cpu_scalar, cpu_simd, blas, cuda };

pub const MatmulOpts = struct {
    backend: Backend = .auto,
    trans_a: bool = false,
    trans_b: bool = false,
    alpha: f64 = 1.0,
    beta: f64 = 0.0,
};

pub const OpKind = enum {
    mat_alloc,
    vec_alloc,
    clone,
    add,
    sub,
    mul,
    div,
    relu,
    sum,
    max,
    dot,
    norm2,
    matmul,
    frame_reset,
    frame_mat_alloc,
    frame_vec_alloc,
};

pub const Failure = struct {
    op: OpKind,
    backend: Backend,
    err: LaError,
};

pub fn makeFailure(op: OpKind, backend: Backend, err: LaError) Failure {
    return .{
        .op = op,
        .backend = backend,
        .err = err,
    };
}

pub fn Outcome(comptime T: type) type {
    return union(enum) {
        ok: T,
        fail: Failure,
    };
}

pub fn intoOutcome(comptime T: type, op: OpKind, backend: Backend, result: LaError!T) Outcome(T) {
    return if (result) |ok| .{ .ok = ok } else |err| .{
        .fail = .{
            .op = op,
            .backend = backend,
            .err = err,
        },
    };
}

pub fn unwrapOutcome(comptime T: type, result: Outcome(T)) LaError!T {
    return switch (result) {
        .ok => |ok| ok,
        .fail => |fail| fail.err,
    };
}

pub fn failOutcome(comptime T: type, op: OpKind, backend: Backend, err: LaError) Outcome(T) {
    return .{ .fail = makeFailure(op, backend, err) };
}

pub fn assertNumericType(comptime T: type) void {
    switch (@typeInfo(T)) {
        .int, .comptime_int, .float, .comptime_float => {},
        else => @compileError("unsupported scalar type: " ++ @typeName(T)),
    }
}

pub fn isFloatType(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .float, .comptime_float => true,
        else => false,
    };
}

pub fn isSignedIntType(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int => |int_info| int_info.signedness == .signed,
        .comptime_int => true,
        else => false,
    };
}

pub fn ScalarOf(comptime X: type) type {
    const Base = stripPointer(X);
    if (hasDeclSafe(Base, "Scalar")) return Base.Scalar;
    @compileError("type does not expose Scalar: " ++ @typeName(X));
}

pub fn mulChecked(a: usize, b: usize) LaError!usize {
    return std.math.mul(usize, a, b) catch error.DimensionMismatch;
}

pub fn addChecked(a: usize, b: usize) LaError!usize {
    return std.math.add(usize, a, b) catch error.DimensionMismatch;
}

fn stripPointer(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .pointer => |ptr| if (ptr.size == .one) stripPointer(ptr.child) else T,
        else => T,
    };
}

fn hasDeclSafe(comptime T: type, comptime name: []const u8) bool {
    return switch (@typeInfo(T)) {
        .@"struct", .@"enum", .@"union", .@"opaque" => @hasDecl(T, name),
        else => false,
    };
}
