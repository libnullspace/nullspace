const std = @import("std");
const core = @import("core.zig");
const lview = @import("view.zig");

pub fn Matrix(comptime T: type) type {
    core.assertNumericType(T);
    return struct {
        pub const Scalar = T;

        data: []T,
        rows: usize,
        cols: usize,

        const Self = @This();

        pub fn init(alloc: std.mem.Allocator, rows: usize, cols: usize) core.LaError!Self {
            const count = try core.mulChecked(rows, cols);
            const data = try alloc.alloc(T, count);
            return .{ .data = data, .rows = rows, .cols = cols };
        }

        // Caller must deinit with the allocator that originally allocated `owned`.
        pub fn fromOwnedSliceAssumeAllocator(owned: []T, rows: usize, cols: usize) core.LaError!Self {
            const expected = try core.mulChecked(rows, cols);
            if (owned.len != expected) return error.DimensionMismatch;
            return .{ .data = owned, .rows = rows, .cols = cols };
        }

        pub fn fromOwnedSlice(owned: []T, rows: usize, cols: usize) core.LaError!Self {
            return fromOwnedSliceAssumeAllocator(owned, rows, cols);
        }

        pub fn fromUnmanagedArrayList(
            alloc: std.mem.Allocator,
            list: *std.ArrayList(T),
            rows: usize,
            cols: usize,
        ) core.LaError!Self {
            const expected = try core.mulChecked(rows, cols);
            if (list.items.len != expected) return error.DimensionMismatch;
            return .{
                .data = try list.toOwnedSlice(alloc),
                .rows = rows,
                .cols = cols,
            };
        }

        pub fn fromManagedArrayList(
            list: *std.array_list.Managed(T),
            rows: usize,
            cols: usize,
        ) core.LaError!Self {
            const expected = try core.mulChecked(rows, cols);
            if (list.items.len != expected) return error.DimensionMismatch;
            return .{
                .data = try list.toOwnedSlice(),
                .rows = rows,
                .cols = cols,
            };
        }

        pub fn deinit(self: *Self, alloc: std.mem.Allocator) void {
            alloc.free(self.data);
            self.data = self.data[0..0];
            self.rows = 0;
            self.cols = 0;
        }

        pub fn toOwnedSlice(self: *Self) []T {
            const owned = self.data;
            self.data = self.data[0..0];
            self.rows = 0;
            self.cols = 0;
            return owned;
        }

        pub fn toUnmanagedArrayList(self: *Self) std.ArrayList(T) {
            return std.ArrayList(T).fromOwnedSlice(self.toOwnedSlice());
        }

        pub fn toManagedArrayList(self: *Self, alloc: std.mem.Allocator) std.array_list.Managed(T) {
            return std.array_list.Managed(T).fromOwnedSlice(alloc, self.toOwnedSlice());
        }

        pub fn fill(self: Self, value: T) void {
            for (self.data) |*item| {
                item.* = value;
            }
        }

        pub fn asMatView(self: Self) lview.MatView(T) {
            return .{
                .ptr = self.data.ptr,
                .rows = self.rows,
                .cols = self.cols,
                .row_stride = self.cols,
                .col_stride = 1,
            };
        }

        pub fn asMatMutView(self: Self) lview.MatMutView(T) {
            return .{
                .ptr = self.data.ptr,
                .rows = self.rows,
                .cols = self.cols,
                .row_stride = self.cols,
                .col_stride = 1,
            };
        }

        pub fn view(self: Self) lview.MatView(T) {
            return self.asMatView();
        }

        pub fn viewMut(self: Self) lview.MatMutView(T) {
            return self.asMatMutView();
        }
    };
}

pub fn Vector(comptime T: type) type {
    core.assertNumericType(T);
    return struct {
        pub const Scalar = T;

        data: []T,
        len: usize,

        const Self = @This();

        pub fn init(alloc: std.mem.Allocator, len: usize) core.LaError!Self {
            const data = try alloc.alloc(T, len);
            return .{ .data = data, .len = len };
        }

        // Caller must deinit with the allocator that originally allocated `owned`.
        pub fn fromOwnedSliceAssumeAllocator(owned: []T) Self {
            return .{ .data = owned, .len = owned.len };
        }

        pub fn fromOwnedSlice(owned: []T) Self {
            return fromOwnedSliceAssumeAllocator(owned);
        }

        pub fn fromUnmanagedArrayList(alloc: std.mem.Allocator, list: *std.ArrayList(T)) core.LaError!Self {
            const len = list.items.len;
            return .{ .data = try list.toOwnedSlice(alloc), .len = len };
        }

        pub fn fromManagedArrayList(list: *std.array_list.Managed(T)) core.LaError!Self {
            const len = list.items.len;
            return .{ .data = try list.toOwnedSlice(), .len = len };
        }

        pub fn deinit(self: *Self, alloc: std.mem.Allocator) void {
            alloc.free(self.data);
            self.data = self.data[0..0];
            self.len = 0;
        }

        pub fn toOwnedSlice(self: *Self) []T {
            const owned = self.data;
            self.data = self.data[0..0];
            self.len = 0;
            return owned;
        }

        pub fn toUnmanagedArrayList(self: *Self) std.ArrayList(T) {
            return std.ArrayList(T).fromOwnedSlice(self.toOwnedSlice());
        }

        pub fn toManagedArrayList(self: *Self, alloc: std.mem.Allocator) std.array_list.Managed(T) {
            return std.array_list.Managed(T).fromOwnedSlice(alloc, self.toOwnedSlice());
        }

        pub fn fill(self: Self, value: T) void {
            for (self.data) |*item| {
                item.* = value;
            }
        }

        pub fn asVecView(self: Self) lview.VecView(T) {
            return .{ .ptr = self.data.ptr, .len = self.len, .stride = 1 };
        }

        pub fn asVecMutView(self: Self) lview.VecMutView(T) {
            return .{ .ptr = self.data.ptr, .len = self.len, .stride = 1 };
        }

        pub fn view(self: Self) lview.VecView(T) {
            return self.asVecView();
        }

        pub fn viewMut(self: Self) lview.VecMutView(T) {
            return self.asVecMutView();
        }
    };
}

fn TempMatImpl(comptime T: type) type {
    return struct {
        data: []T,
        rows: usize,
        cols: usize,
    };
}

fn tempMatImpl(comptime T: type, temp: *TempMat(T)) *TempMatImpl(T) {
    return @ptrCast(@alignCast(temp));
}

fn tempMatImplConst(comptime T: type, temp: *const TempMat(T)) *const TempMatImpl(T) {
    return @ptrCast(@alignCast(temp));
}

pub fn allocTempMat(alloc: std.mem.Allocator, comptime T: type, rows: usize, cols: usize) core.LaError!*TempMat(T) {
    const count = try core.mulChecked(rows, cols);
    const data = try alloc.alloc(T, count);
    errdefer alloc.free(data);

    const temp = try alloc.create(TempMatImpl(T));
    temp.* = .{
        .data = data,
        .rows = rows,
        .cols = cols,
    };
    return @ptrCast(temp);
}

pub fn TempMat(comptime T: type) type {
    core.assertNumericType(T);
    return opaque {
        pub const Scalar = T;

        const Self = @This();

        pub fn fill(self: *Self, value: T) void {
            const temp = tempMatImpl(T, self);
            for (temp.data) |*item| {
                item.* = value;
            }
        }

        pub fn asMatView(self: *const Self) lview.MatView(T) {
            const temp = tempMatImplConst(T, self);
            return .{
                .ptr = temp.data.ptr,
                .rows = temp.rows,
                .cols = temp.cols,
                .row_stride = temp.cols,
                .col_stride = 1,
            };
        }

        pub fn asMatMutView(self: *Self) lview.MatMutView(T) {
            const temp = tempMatImpl(T, self);
            return .{
                .ptr = temp.data.ptr,
                .rows = temp.rows,
                .cols = temp.cols,
                .row_stride = temp.cols,
                .col_stride = 1,
            };
        }

        pub fn view(self: *const Self) lview.MatView(T) {
            return self.asMatView();
        }

        pub fn viewMut(self: *Self) lview.MatMutView(T) {
            return self.asMatMutView();
        }
    };
}

fn TempVecImpl(comptime T: type) type {
    return struct {
        data: []T,
        len: usize,
    };
}

fn tempVecImpl(comptime T: type, temp: *TempVec(T)) *TempVecImpl(T) {
    return @ptrCast(@alignCast(temp));
}

fn tempVecImplConst(comptime T: type, temp: *const TempVec(T)) *const TempVecImpl(T) {
    return @ptrCast(@alignCast(temp));
}

pub fn allocTempVec(alloc: std.mem.Allocator, comptime T: type, len: usize) core.LaError!*TempVec(T) {
    const data = try alloc.alloc(T, len);
    errdefer alloc.free(data);

    const temp = try alloc.create(TempVecImpl(T));
    temp.* = .{
        .data = data,
        .len = len,
    };
    return @ptrCast(temp);
}

pub fn TempVec(comptime T: type) type {
    core.assertNumericType(T);
    return opaque {
        pub const Scalar = T;

        const Self = @This();

        pub fn fill(self: *Self, value: T) void {
            const temp = tempVecImpl(T, self);
            for (temp.data) |*item| {
                item.* = value;
            }
        }

        pub fn asVecView(self: *const Self) lview.VecView(T) {
            const temp = tempVecImplConst(T, self);
            return .{ .ptr = temp.data.ptr, .len = temp.len, .stride = 1 };
        }

        pub fn asVecMutView(self: *Self) lview.VecMutView(T) {
            const temp = tempVecImpl(T, self);
            return .{ .ptr = temp.data.ptr, .len = temp.len, .stride = 1 };
        }

        pub fn view(self: *const Self) lview.VecView(T) {
            return self.asVecView();
        }

        pub fn viewMut(self: *Self) lview.VecMutView(T) {
            return self.asVecMutView();
        }
    };
}
