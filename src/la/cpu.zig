const std = @import("std");
const core = @import("core.zig");
const view = @import("view.zig");
const owning = @import("owning.zig");
const ops = @import("ops.zig");

pub fn cpu(alloc: std.mem.Allocator) Cpu {
    return .{ .alloc = alloc };
}

pub const Cpu = struct {
    alloc: std.mem.Allocator,
    simd: core.SimdMode = .auto,
    threads: core.Threading = .single,

    pub fn deinit(self: *Cpu) void {
        _ = self;
    }

    pub fn mat(self: *Cpu, comptime T: type, rows: usize, cols: usize) core.LaError!owning.Matrix(T) {
        return core.unwrapOutcome(owning.Matrix(T), self.mat_with_failure(T, rows, cols));
    }

    pub fn mat_with_failure(self: *Cpu, comptime T: type, rows: usize, cols: usize) core.Outcome(owning.Matrix(T)) {
        return core.intoOutcome(owning.Matrix(T), .mat_alloc, .cpu_scalar, owning.Matrix(T).init(self.alloc, rows, cols));
    }

    pub fn vec(self: *Cpu, comptime T: type, len: usize) core.LaError!owning.Vector(T) {
        return core.unwrapOutcome(owning.Vector(T), self.vec_with_failure(T, len));
    }

    pub fn vec_with_failure(self: *Cpu, comptime T: type, len: usize) core.Outcome(owning.Vector(T)) {
        return core.intoOutcome(owning.Vector(T), .vec_alloc, .cpu_scalar, owning.Vector(T).init(self.alloc, len));
    }

    pub fn clone(self: *Cpu, x: anytype) core.LaError!owning.Matrix(core.ScalarOf(@TypeOf(x))) {
        const T = core.ScalarOf(@TypeOf(x));
        return core.unwrapOutcome(owning.Matrix(T), self.clone_with_failure(x));
    }

    pub fn clone_with_failure(self: *Cpu, x: anytype) core.Outcome(owning.Matrix(core.ScalarOf(@TypeOf(x)))) {
        const T = core.ScalarOf(@TypeOf(x));
        const xv = view.matViewFromAny(x);
        return core.intoOutcome(owning.Matrix(T), .clone, .cpu_scalar, ops.cloneAlloc(T, self.alloc, xv));
    }

    pub fn frame(self: *Cpu) Frame {
        return Frame.init(self);
    }

    pub fn add(self: *Cpu, a: anytype, b: anytype) core.LaError!owning.Matrix(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(owning.Matrix(T), self.add_with_failure(a, b));
    }

    pub fn add_with_failure(self: *Cpu, a: anytype, b: anytype) core.Outcome(owning.Matrix(core.ScalarOf(@TypeOf(a)))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("add expects matching scalar types");

        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);
        return core.intoOutcome(owning.Matrix(T), .add, .cpu_scalar, ops.addAlloc(T, self.alloc, av, bv));
    }

    pub fn sub(self: *Cpu, a: anytype, b: anytype) core.LaError!owning.Matrix(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(owning.Matrix(T), self.sub_with_failure(a, b));
    }

    pub fn sub_with_failure(self: *Cpu, a: anytype, b: anytype) core.Outcome(owning.Matrix(core.ScalarOf(@TypeOf(a)))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("sub expects matching scalar types");

        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);
        return core.intoOutcome(owning.Matrix(T), .sub, .cpu_scalar, ops.subAlloc(T, self.alloc, av, bv));
    }

    pub fn mul(self: *Cpu, a: anytype, b: anytype) core.LaError!owning.Matrix(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(owning.Matrix(T), self.mul_with_failure(a, b));
    }

    pub fn mul_with_failure(self: *Cpu, a: anytype, b: anytype) core.Outcome(owning.Matrix(core.ScalarOf(@TypeOf(a)))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("mul expects matching scalar types");

        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);
        return core.intoOutcome(owning.Matrix(T), .mul, .cpu_scalar, ops.mulAlloc(T, self.alloc, av, bv));
    }

    pub fn div(self: *Cpu, a: anytype, b: anytype) core.LaError!owning.Matrix(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(owning.Matrix(T), self.div_with_failure(a, b));
    }

    pub fn div_with_failure(self: *Cpu, a: anytype, b: anytype) core.Outcome(owning.Matrix(core.ScalarOf(@TypeOf(a)))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("div expects matching scalar types");

        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);
        return core.intoOutcome(owning.Matrix(T), .div, .cpu_scalar, ops.divAlloc(T, self.alloc, av, bv));
    }

    pub fn relu(self: *Cpu, x: anytype) core.LaError!owning.Matrix(core.ScalarOf(@TypeOf(x))) {
        const T = core.ScalarOf(@TypeOf(x));
        return core.unwrapOutcome(owning.Matrix(T), self.relu_with_failure(x));
    }

    pub fn relu_with_failure(self: *Cpu, x: anytype) core.Outcome(owning.Matrix(core.ScalarOf(@TypeOf(x)))) {
        const T = core.ScalarOf(@TypeOf(x));
        const xv = view.matViewFromAny(x);
        return core.intoOutcome(owning.Matrix(T), .relu, .cpu_scalar, ops.reluAlloc(T, self.alloc, xv));
    }

    pub fn matmul(self: *Cpu, a: anytype, b: anytype) core.LaError!owning.Matrix(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(owning.Matrix(T), self.matmul_with_failure(a, b));
    }

    pub fn matmul_with_failure(self: *Cpu, a: anytype, b: anytype) core.Outcome(owning.Matrix(core.ScalarOf(@TypeOf(a)))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("matmul expects matching scalar types");

        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);
        const backend = self.matmulBackend();
        return core.intoOutcome(owning.Matrix(T), .matmul, backend, ops.matmulAlloc(T, self.alloc, av, bv, .{ .backend = backend }));
    }

    pub fn add_into(self: *Cpu, out: anytype, a: anytype, b: anytype) core.LaError!void {
        return core.unwrapOutcome(void, self.add_into_with_failure(out, a, b));
    }

    pub fn add_into_with_failure(self: *Cpu, out: anytype, a: anytype, b: anytype) core.Outcome(void) {
        const backend = self.elementwiseBackend();
        const result: core.LaError!void = blk: {
            if (self.simd == .simd) break :blk error.BackendUnavailable;

            const OutT = core.ScalarOf(@TypeOf(out));
            const A = core.ScalarOf(@TypeOf(a));
            const B = core.ScalarOf(@TypeOf(b));
            if (OutT != A or A != B) @compileError("add_into expects matching scalar types");

            const outv = view.matMutViewFromAny(out);
            const av = view.matViewFromAny(a);
            const bv = view.matViewFromAny(b);
            break :blk ops.addInto(OutT, outv, av, bv);
        };
        return core.intoOutcome(void, .add, backend, result);
    }

    pub fn sub_into(self: *Cpu, out: anytype, a: anytype, b: anytype) core.LaError!void {
        return core.unwrapOutcome(void, self.sub_into_with_failure(out, a, b));
    }

    pub fn sub_into_with_failure(self: *Cpu, out: anytype, a: anytype, b: anytype) core.Outcome(void) {
        const backend = self.elementwiseBackend();
        const result: core.LaError!void = blk: {
            if (self.simd == .simd) break :blk error.BackendUnavailable;

            const OutT = core.ScalarOf(@TypeOf(out));
            const A = core.ScalarOf(@TypeOf(a));
            const B = core.ScalarOf(@TypeOf(b));
            if (OutT != A or A != B) @compileError("sub_into expects matching scalar types");

            const outv = view.matMutViewFromAny(out);
            const av = view.matViewFromAny(a);
            const bv = view.matViewFromAny(b);
            break :blk ops.subInto(OutT, outv, av, bv);
        };
        return core.intoOutcome(void, .sub, backend, result);
    }

    pub fn mul_into(self: *Cpu, out: anytype, a: anytype, b: anytype) core.LaError!void {
        return core.unwrapOutcome(void, self.mul_into_with_failure(out, a, b));
    }

    pub fn mul_into_with_failure(self: *Cpu, out: anytype, a: anytype, b: anytype) core.Outcome(void) {
        const backend = self.elementwiseBackend();
        const result: core.LaError!void = blk: {
            if (self.simd == .simd) break :blk error.BackendUnavailable;

            const OutT = core.ScalarOf(@TypeOf(out));
            const A = core.ScalarOf(@TypeOf(a));
            const B = core.ScalarOf(@TypeOf(b));
            if (OutT != A or A != B) @compileError("mul_into expects matching scalar types");

            const outv = view.matMutViewFromAny(out);
            const av = view.matViewFromAny(a);
            const bv = view.matViewFromAny(b);
            break :blk ops.mulInto(OutT, outv, av, bv);
        };
        return core.intoOutcome(void, .mul, backend, result);
    }

    pub fn div_into(self: *Cpu, out: anytype, a: anytype, b: anytype) core.LaError!void {
        return core.unwrapOutcome(void, self.div_into_with_failure(out, a, b));
    }

    pub fn div_into_with_failure(self: *Cpu, out: anytype, a: anytype, b: anytype) core.Outcome(void) {
        const backend = self.elementwiseBackend();
        const result: core.LaError!void = blk: {
            if (self.simd == .simd) break :blk error.BackendUnavailable;

            const OutT = core.ScalarOf(@TypeOf(out));
            const A = core.ScalarOf(@TypeOf(a));
            const B = core.ScalarOf(@TypeOf(b));
            if (OutT != A or A != B) @compileError("div_into expects matching scalar types");

            const outv = view.matMutViewFromAny(out);
            const av = view.matViewFromAny(a);
            const bv = view.matViewFromAny(b);
            break :blk ops.divInto(OutT, outv, av, bv);
        };
        return core.intoOutcome(void, .div, backend, result);
    }

    pub fn matmul_into(self: *Cpu, out: anytype, a: anytype, b: anytype) core.LaError!void {
        return core.unwrapOutcome(void, self.matmul_into_with_failure(out, a, b));
    }

    pub fn matmul_into_with_failure(self: *Cpu, out: anytype, a: anytype, b: anytype) core.Outcome(void) {
        const backend = self.matmulBackend();
        const result: core.LaError!void = blk: {
            const OutT = core.ScalarOf(@TypeOf(out));
            const A = core.ScalarOf(@TypeOf(a));
            const B = core.ScalarOf(@TypeOf(b));
            if (OutT != A or A != B) @compileError("matmul_into expects matching scalar types");

            const outv = view.matMutViewFromAny(out);
            const av = view.matViewFromAny(a);
            const bv = view.matViewFromAny(b);
            break :blk ops.matmulInto(OutT, outv, av, bv, .{ .backend = backend });
        };
        return core.intoOutcome(void, .matmul, backend, result);
    }

    pub fn relu_inplace(self: *Cpu, x: anytype) core.LaError!void {
        return core.unwrapOutcome(void, self.relu_inplace_with_failure(x));
    }

    pub fn relu_inplace_with_failure(self: *Cpu, x: anytype) core.Outcome(void) {
        const backend = self.elementwiseBackend();
        const result: core.LaError!void = blk: {
            if (self.simd == .simd) break :blk error.BackendUnavailable;

            const T = core.ScalarOf(@TypeOf(x));
            const xv = view.matMutViewFromAny(x);
            break :blk ops.reluInplace(T, xv);
        };
        return core.intoOutcome(void, .relu, backend, result);
    }

    pub fn sum(self: *Cpu, x: anytype) core.LaError!core.ScalarOf(@TypeOf(x)) {
        const T = core.ScalarOf(@TypeOf(x));
        return core.unwrapOutcome(T, self.sum_with_failure(x));
    }

    pub fn sum_with_failure(self: *Cpu, x: anytype) core.Outcome(core.ScalarOf(@TypeOf(x))) {
        const T = core.ScalarOf(@TypeOf(x));
        const backend = self.elementwiseBackend();
        return core.intoOutcome(T, .sum, backend, ops.sum(T, view.matViewFromAny(x)));
    }

    pub fn max(self: *Cpu, x: anytype) core.LaError!core.ScalarOf(@TypeOf(x)) {
        const T = core.ScalarOf(@TypeOf(x));
        return core.unwrapOutcome(T, self.max_with_failure(x));
    }

    pub fn max_with_failure(self: *Cpu, x: anytype) core.Outcome(core.ScalarOf(@TypeOf(x))) {
        const T = core.ScalarOf(@TypeOf(x));
        const backend = self.elementwiseBackend();
        return core.intoOutcome(T, .max, backend, ops.max(T, view.matViewFromAny(x)));
    }

    pub fn norm2(self: *Cpu, x: anytype) core.LaError!core.ScalarOf(@TypeOf(x)) {
        const T = core.ScalarOf(@TypeOf(x));
        return core.unwrapOutcome(T, self.norm2_with_failure(x));
    }

    pub fn norm2_with_failure(self: *Cpu, x: anytype) core.Outcome(core.ScalarOf(@TypeOf(x))) {
        const T = core.ScalarOf(@TypeOf(x));
        const backend = self.elementwiseBackend();
        return core.intoOutcome(T, .norm2, backend, ops.norm2(T, view.matViewFromAny(x)));
    }

    pub fn dot(self: *Cpu, a: anytype, b: anytype) core.LaError!core.ScalarOf(@TypeOf(a)) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(T, self.dot_with_failure(a, b));
    }

    pub fn dot_with_failure(self: *Cpu, a: anytype, b: anytype) core.Outcome(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("dot expects matching scalar types");

        const backend = self.elementwiseBackend();
        return core.intoOutcome(T, .dot, backend, ops.dot(T, view.vecViewFromAny(a), view.vecViewFromAny(b)));
    }

    fn elementwiseBackend(self: *Cpu) core.Backend {
        return switch (self.simd) {
            .simd => .cpu_simd,
            .auto, .scalar => .cpu_scalar,
        };
    }

    fn matmulBackend(self: *Cpu) core.Backend {
        return switch (self.simd) {
            .auto => .auto,
            .scalar => .cpu_scalar,
            .simd => .cpu_simd,
        };
    }
};

pub const Frame = struct {
    cpu: *Cpu,
    arena: std.heap.ArenaAllocator,

    pub fn init(cpu_ref: *Cpu) Frame {
        return .{
            .cpu = cpu_ref,
            .arena = std.heap.ArenaAllocator.init(cpu_ref.alloc),
        };
    }

    pub fn deinit(self: *Frame) void {
        self.arena.deinit();
    }

    pub fn reset(self: *Frame) core.LaError!void {
        return core.unwrapOutcome(void, self.reset_with_failure());
    }

    pub fn reset_with_failure(self: *Frame) core.Outcome(void) {
        const result: core.LaError!void = blk: {
            if (!self.arena.reset(.retain_capacity)) break :blk error.OutOfMemory;
            break :blk {};
        };
        return core.intoOutcome(void, .frame_reset, .cpu_scalar, result);
    }

    pub fn mat(self: *Frame, comptime T: type, rows: usize, cols: usize) core.LaError!*owning.TempMat(T) {
        return core.unwrapOutcome(*owning.TempMat(T), self.mat_with_failure(T, rows, cols));
    }

    pub fn mat_with_failure(self: *Frame, comptime T: type, rows: usize, cols: usize) core.Outcome(*owning.TempMat(T)) {
        return core.intoOutcome(*owning.TempMat(T), .frame_mat_alloc, .cpu_scalar, self.allocTempMat(T, rows, cols));
    }

    pub fn vec(self: *Frame, comptime T: type, len: usize) core.LaError!*owning.TempVec(T) {
        return core.unwrapOutcome(*owning.TempVec(T), self.vec_with_failure(T, len));
    }

    pub fn vec_with_failure(self: *Frame, comptime T: type, len: usize) core.Outcome(*owning.TempVec(T)) {
        return core.intoOutcome(*owning.TempVec(T), .frame_vec_alloc, .cpu_scalar, self.allocTempVec(T, len));
    }

    pub fn add(self: *Frame, a: anytype, b: anytype) core.LaError!*owning.TempMat(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(*owning.TempMat(T), self.add_with_failure(a, b));
    }

    pub fn add_with_failure(self: *Frame, a: anytype, b: anytype) core.Outcome(*owning.TempMat(core.ScalarOf(@TypeOf(a)))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("frame.add expects matching scalar types");

        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);

        const result: core.LaError!*owning.TempMat(T) = blk: {
            var out = try self.allocTempMat(T, av.rows, av.cols);
            try ops.addInto(T, out.asMatMutView(), av, bv);
            break :blk out;
        };
        return core.intoOutcome(*owning.TempMat(T), .add, .cpu_scalar, result);
    }

    pub fn sub(self: *Frame, a: anytype, b: anytype) core.LaError!*owning.TempMat(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(*owning.TempMat(T), self.sub_with_failure(a, b));
    }

    pub fn sub_with_failure(self: *Frame, a: anytype, b: anytype) core.Outcome(*owning.TempMat(core.ScalarOf(@TypeOf(a)))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("frame.sub expects matching scalar types");

        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);

        const result: core.LaError!*owning.TempMat(T) = blk: {
            var out = try self.allocTempMat(T, av.rows, av.cols);
            try ops.subInto(T, out.asMatMutView(), av, bv);
            break :blk out;
        };
        return core.intoOutcome(*owning.TempMat(T), .sub, .cpu_scalar, result);
    }

    pub fn mul(self: *Frame, a: anytype, b: anytype) core.LaError!*owning.TempMat(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(*owning.TempMat(T), self.mul_with_failure(a, b));
    }

    pub fn mul_with_failure(self: *Frame, a: anytype, b: anytype) core.Outcome(*owning.TempMat(core.ScalarOf(@TypeOf(a)))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("frame.mul expects matching scalar types");

        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);

        const result: core.LaError!*owning.TempMat(T) = blk: {
            var out = try self.allocTempMat(T, av.rows, av.cols);
            try ops.mulInto(T, out.asMatMutView(), av, bv);
            break :blk out;
        };
        return core.intoOutcome(*owning.TempMat(T), .mul, .cpu_scalar, result);
    }

    pub fn div(self: *Frame, a: anytype, b: anytype) core.LaError!*owning.TempMat(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(*owning.TempMat(T), self.div_with_failure(a, b));
    }

    pub fn div_with_failure(self: *Frame, a: anytype, b: anytype) core.Outcome(*owning.TempMat(core.ScalarOf(@TypeOf(a)))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("frame.div expects matching scalar types");

        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);

        const result: core.LaError!*owning.TempMat(T) = blk: {
            var out = try self.allocTempMat(T, av.rows, av.cols);
            try ops.divInto(T, out.asMatMutView(), av, bv);
            break :blk out;
        };
        return core.intoOutcome(*owning.TempMat(T), .div, .cpu_scalar, result);
    }

    pub fn relu(self: *Frame, x: anytype) core.LaError!*owning.TempMat(core.ScalarOf(@TypeOf(x))) {
        const T = core.ScalarOf(@TypeOf(x));
        return core.unwrapOutcome(*owning.TempMat(T), self.relu_with_failure(x));
    }

    pub fn relu_with_failure(self: *Frame, x: anytype) core.Outcome(*owning.TempMat(core.ScalarOf(@TypeOf(x)))) {
        const T = core.ScalarOf(@TypeOf(x));
        const xv = view.matViewFromAny(x);
        var out = self.allocTempMat(T, xv.rows, xv.cols) catch |err| {
            return core.failOutcome(*owning.TempMat(T), .relu, .cpu_scalar, err);
        };
        var out_view = out.asMatMutView();

        var i: usize = 0;
        while (i < xv.rows) : (i += 1) {
            var j: usize = 0;
            while (j < xv.cols) : (j += 1) {
                out_view.set(i, j, xv.get(i, j));
            }
        }

        ops.reluInplace(T, out_view) catch |err| {
            return core.failOutcome(*owning.TempMat(T), .relu, .cpu_scalar, err);
        };
        return .{ .ok = out };
    }

    pub fn matmul(self: *Frame, a: anytype, b: anytype) core.LaError!*owning.TempMat(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(*owning.TempMat(T), self.matmul_with_failure(a, b));
    }

    pub fn matmul_with_failure(self: *Frame, a: anytype, b: anytype) core.Outcome(*owning.TempMat(core.ScalarOf(@TypeOf(a)))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("frame.matmul expects matching scalar types");

        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);

        const backend = self.cpu.matmulBackend();
        const opts: core.MatmulOpts = .{ .backend = backend };
        const out_rows = if (opts.trans_a) av.cols else av.rows;
        const out_cols = if (opts.trans_b) bv.rows else bv.cols;

        const result: core.LaError!*owning.TempMat(T) = blk: {
            var out = try self.allocTempMat(T, out_rows, out_cols);
            out.fill(@as(T, 0));
            try ops.matmulInto(T, out.asMatMutView(), av, bv, opts);
            break :blk out;
        };
        return core.intoOutcome(*owning.TempMat(T), .matmul, backend, result);
    }

    fn allocTempMat(self: *Frame, comptime T: type, rows: usize, cols: usize) core.LaError!*owning.TempMat(T) {
        return owning.allocTempMat(self.allocator(), T, rows, cols);
    }

    fn allocTempVec(self: *Frame, comptime T: type, len: usize) core.LaError!*owning.TempVec(T) {
        return owning.allocTempVec(self.allocator(), T, len);
    }

    fn allocator(self: *Frame) std.mem.Allocator {
        return self.arena.allocator();
    }
};
