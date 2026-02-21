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

pub const Exec = struct {
    simd: SimdMode = .auto,
    threads: Threading = .single,
    device: enum { cpu } = .cpu,
};

pub fn GemmOpts(comptime T: type) type {
    assertNumericType(T);
    return struct {
        alpha: T = @as(T, 1),
        beta: T = @as(T, 0),
    };
}

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
    exec: Exec,
    err: LaError,
};

pub fn makeFailure(op: OpKind, exec: Exec, err: LaError) Failure {
    return .{
        .op = op,
        .exec = exec,
        .err = err,
    };
}

pub fn Outcome(comptime T: type) type {
    return union(enum) {
        ok: T,
        fail: Failure,
    };
}

pub fn intoOutcome(comptime T: type, op: OpKind, exec: Exec, result: LaError!T) Outcome(T) {
    return if (result) |ok| .{ .ok = ok } else |err| .{
        .fail = .{
            .op = op,
            .exec = exec,
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

pub fn failOutcome(comptime T: type, op: OpKind, exec: Exec, err: LaError) Outcome(T) {
    return .{ .fail = makeFailure(op, exec, err) };
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
