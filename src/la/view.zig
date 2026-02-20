const std = @import("std");
const core = @import("core.zig");

pub fn VecView(comptime T: type) type {
    core.assertNumericType(T);
    return struct {
        pub const Scalar = T;

        ptr: [*]const T,
        len: usize,
        stride: usize = 1,

        const Self = @This();

        pub fn fromSlice(slice: []const T) Self {
            return .{ .ptr = slice.ptr, .len = slice.len, .stride = 1 };
        }

        pub fn at(self: Self, index: usize) T {
            std.debug.assert(index < self.len);
            return self.ptr[index * self.stride];
        }

        pub fn isContiguous(self: Self) bool {
            return self.stride == 1;
        }
    };
}

pub fn VecMutView(comptime T: type) type {
    core.assertNumericType(T);
    return struct {
        pub const Scalar = T;

        ptr: [*]T,
        len: usize,
        stride: usize = 1,

        const Self = @This();

        pub fn fromSlice(slice: []T) Self {
            return .{ .ptr = slice.ptr, .len = slice.len, .stride = 1 };
        }

        pub fn at(self: Self, index: usize) T {
            std.debug.assert(index < self.len);
            return self.ptr[index * self.stride];
        }

        pub fn set(self: Self, index: usize, value: T) void {
            std.debug.assert(index < self.len);
            self.ptr[index * self.stride] = value;
        }

        pub fn asConst(self: Self) VecView(T) {
            return .{ .ptr = self.ptr, .len = self.len, .stride = self.stride };
        }

        pub fn isContiguous(self: Self) bool {
            return self.stride == 1;
        }
    };
}

pub fn MatView(comptime T: type) type {
    core.assertNumericType(T);
    return struct {
        pub const Scalar = T;

        ptr: [*]const T,
        rows: usize,
        cols: usize,
        row_stride: usize,
        col_stride: usize,

        const Self = @This();

        pub fn fromSliceRowMajor(slice: []const T, rows: usize, cols: usize) core.LaError!Self {
            const expected = try core.mulChecked(rows, cols);
            if (slice.len != expected) return error.DimensionMismatch;
            return .{
                .ptr = slice.ptr,
                .rows = rows,
                .cols = cols,
                .row_stride = cols,
                .col_stride = 1,
            };
        }

        pub fn get(self: Self, row: usize, col: usize) T {
            std.debug.assert(row < self.rows);
            std.debug.assert(col < self.cols);
            return self.ptr[row * self.row_stride + col * self.col_stride];
        }

        pub fn isContiguousRowMajor(self: Self) bool {
            return self.col_stride == 1 and self.row_stride == self.cols;
        }
    };
}

pub fn MatMutView(comptime T: type) type {
    core.assertNumericType(T);
    return struct {
        pub const Scalar = T;

        ptr: [*]T,
        rows: usize,
        cols: usize,
        row_stride: usize,
        col_stride: usize,

        const Self = @This();

        pub fn fromSliceRowMajor(slice: []T, rows: usize, cols: usize) core.LaError!Self {
            const expected = try core.mulChecked(rows, cols);
            if (slice.len != expected) return error.DimensionMismatch;
            return .{
                .ptr = slice.ptr,
                .rows = rows,
                .cols = cols,
                .row_stride = cols,
                .col_stride = 1,
            };
        }

        pub fn get(self: Self, row: usize, col: usize) T {
            std.debug.assert(row < self.rows);
            std.debug.assert(col < self.cols);
            return self.ptr[row * self.row_stride + col * self.col_stride];
        }

        pub fn set(self: Self, row: usize, col: usize, value: T) void {
            std.debug.assert(row < self.rows);
            std.debug.assert(col < self.cols);
            self.ptr[row * self.row_stride + col * self.col_stride] = value;
        }

        pub fn asConst(self: Self) MatView(T) {
            return .{
                .ptr = self.ptr,
                .rows = self.rows,
                .cols = self.cols,
                .row_stride = self.row_stride,
                .col_stride = self.col_stride,
            };
        }

        pub fn isContiguousRowMajor(self: Self) bool {
            return self.col_stride == 1 and self.row_stride == self.cols;
        }
    };
}

pub fn matViewFromAny(x: anytype) MatView(core.ScalarOf(@TypeOf(x))) {
    const T = core.ScalarOf(@TypeOf(x));
    const X = @TypeOf(x);

    if (comptime X == MatView(T)) return x;
    if (comptime X == MatMutView(T)) return x.asConst();

    if (comptime hasDeclSafe(X, "asMatView")) return x.asMatView();
    if (comptime hasPointerChildDeclSafe(X, "asMatView")) return x.asMatView();
    if (comptime hasDeclSafe(X, "view")) return x.view();
    if (comptime hasPointerChildDeclSafe(X, "view")) return x.view();
    if (comptime hasFieldSafe(X, "data") and hasFieldSafe(X, "rows") and hasFieldSafe(X, "cols")) {
        return .{
            .ptr = x.data.ptr,
            .rows = x.rows,
            .cols = x.cols,
            .row_stride = x.cols,
            .col_stride = 1,
        };
    }

    switch (@typeInfo(X)) {
        .pointer => |ptr| {
            if (comptime ptr.size == .one) {
                if (comptime ptr.child == MatView(T)) return x.*;
                if (comptime ptr.child == MatMutView(T)) return x.*.asConst();
                if (comptime hasFieldSafe(ptr.child, "data") and hasFieldSafe(ptr.child, "rows") and hasFieldSafe(ptr.child, "cols")) {
                    return .{
                        .ptr = x.*.data.ptr,
                        .rows = x.*.rows,
                        .cols = x.*.cols,
                        .row_stride = x.*.cols,
                        .col_stride = 1,
                    };
                }
            }
        },
        else => {},
    }

    @compileError("cannot convert to MatView: " ++ @typeName(X));
}

pub fn matMutViewFromAny(x: anytype) MatMutView(core.ScalarOf(@TypeOf(x))) {
    const T = core.ScalarOf(@TypeOf(x));
    const X = @TypeOf(x);

    if (comptime X == MatMutView(T)) return x;

    if (comptime hasDeclSafe(X, "asMatMutView")) return x.asMatMutView();
    if (comptime hasPointerChildDeclSafe(X, "asMatMutView")) return x.asMatMutView();
    if (comptime hasDeclSafe(X, "viewMut")) return x.viewMut();
    if (comptime hasPointerChildDeclSafe(X, "viewMut")) return x.viewMut();
    if (comptime hasFieldSafe(X, "data") and hasFieldSafe(X, "rows") and hasFieldSafe(X, "cols")) {
        return .{
            .ptr = x.data.ptr,
            .rows = x.rows,
            .cols = x.cols,
            .row_stride = x.cols,
            .col_stride = 1,
        };
    }

    switch (@typeInfo(X)) {
        .pointer => |ptr| {
            if (comptime ptr.size == .one) {
                if (comptime ptr.child == MatMutView(T)) return x.*;
                if (comptime hasFieldSafe(ptr.child, "data") and hasFieldSafe(ptr.child, "rows") and hasFieldSafe(ptr.child, "cols")) {
                    return .{
                        .ptr = x.*.data.ptr,
                        .rows = x.*.rows,
                        .cols = x.*.cols,
                        .row_stride = x.*.cols,
                        .col_stride = 1,
                    };
                }
            }
        },
        else => {},
    }

    @compileError("cannot convert to MatMutView: " ++ @typeName(X));
}

pub fn vecViewFromAny(x: anytype) VecView(core.ScalarOf(@TypeOf(x))) {
    const T = core.ScalarOf(@TypeOf(x));
    const X = @TypeOf(x);

    if (comptime X == VecView(T)) return x;
    if (comptime X == VecMutView(T)) return x.asConst();

    if (comptime hasDeclSafe(X, "asVecView")) return x.asVecView();
    if (comptime hasPointerChildDeclSafe(X, "asVecView")) return x.asVecView();
    if (comptime hasDeclSafe(X, "view")) return x.view();
    if (comptime hasPointerChildDeclSafe(X, "view")) return x.view();
    if (comptime hasFieldSafe(X, "data") and hasFieldSafe(X, "len")) {
        return .{
            .ptr = x.data.ptr,
            .len = x.len,
            .stride = 1,
        };
    }

    switch (@typeInfo(X)) {
        .pointer => |ptr| {
            if (comptime ptr.size == .one) {
                if (comptime ptr.child == VecView(T)) return x.*;
                if (comptime ptr.child == VecMutView(T)) return x.*.asConst();
                if (comptime hasFieldSafe(ptr.child, "data") and hasFieldSafe(ptr.child, "len")) {
                    return .{
                        .ptr = x.*.data.ptr,
                        .len = x.*.len,
                        .stride = 1,
                    };
                }
            }
        },
        else => {},
    }

    @compileError("cannot convert to VecView: " ++ @typeName(X));
}

pub fn vecMutViewFromAny(x: anytype) VecMutView(core.ScalarOf(@TypeOf(x))) {
    const T = core.ScalarOf(@TypeOf(x));
    const X = @TypeOf(x);

    if (comptime X == VecMutView(T)) return x;

    if (comptime hasDeclSafe(X, "asVecMutView")) return x.asVecMutView();
    if (comptime hasPointerChildDeclSafe(X, "asVecMutView")) return x.asVecMutView();
    if (comptime hasDeclSafe(X, "viewMut")) return x.viewMut();
    if (comptime hasPointerChildDeclSafe(X, "viewMut")) return x.viewMut();
    if (comptime hasFieldSafe(X, "data") and hasFieldSafe(X, "len")) {
        return .{
            .ptr = x.data.ptr,
            .len = x.len,
            .stride = 1,
        };
    }

    switch (@typeInfo(X)) {
        .pointer => |ptr| {
            if (comptime ptr.size == .one) {
                if (comptime ptr.child == VecMutView(T)) return x.*;
                if (comptime hasFieldSafe(ptr.child, "data") and hasFieldSafe(ptr.child, "len")) {
                    return .{
                        .ptr = x.*.data.ptr,
                        .len = x.*.len,
                        .stride = 1,
                    };
                }
            }
        },
        else => {},
    }

    @compileError("cannot convert to VecMutView: " ++ @typeName(X));
}

fn hasDeclSafe(comptime T: type, comptime name: []const u8) bool {
    return switch (@typeInfo(T)) {
        .@"struct", .@"enum", .@"union", .@"opaque" => @hasDecl(T, name),
        else => false,
    };
}

fn hasPointerChildDeclSafe(comptime T: type, comptime name: []const u8) bool {
    return switch (@typeInfo(T)) {
        .pointer => |ptr| ptr.size == .one and hasDeclSafe(ptr.child, name),
        else => false,
    };
}

fn hasFieldSafe(comptime T: type, comptime name: []const u8) bool {
    return switch (@typeInfo(T)) {
        .@"struct" => @hasField(T, name),
        else => false,
    };
}
