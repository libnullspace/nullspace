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

    const alloc_exec = core.Exec{};

    pub fn deinit(self: *Cpu) void {
        _ = self;
    }

    fn exec(self: *Cpu) core.Exec {
        return .{ .simd = self.simd, .threads = self.threads };
    }

    pub fn mat(self: *Cpu, comptime T: type, rows: usize, cols: usize) core.LaError!owning.Matrix(T) {
        return core.unwrapOutcome(owning.Matrix(T), self.mat_with_failure(T, rows, cols));
    }

    pub fn mat_with_failure(self: *Cpu, comptime T: type, rows: usize, cols: usize) core.Outcome(owning.Matrix(T)) {
        return core.intoOutcome(owning.Matrix(T), .mat_alloc, alloc_exec, owning.Matrix(T).init(self.alloc, rows, cols));
    }

    pub fn vec(self: *Cpu, comptime T: type, len: usize) core.LaError!owning.Vector(T) {
        return core.unwrapOutcome(owning.Vector(T), self.vec_with_failure(T, len));
    }

    pub fn vec_with_failure(self: *Cpu, comptime T: type, len: usize) core.Outcome(owning.Vector(T)) {
        return core.intoOutcome(owning.Vector(T), .vec_alloc, alloc_exec, owning.Vector(T).init(self.alloc, len));
    }

    pub fn clone(self: *Cpu, x: anytype) core.LaError!owning.Matrix(core.ScalarOf(@TypeOf(x))) {
        const T = core.ScalarOf(@TypeOf(x));
        return core.unwrapOutcome(owning.Matrix(T), self.clone_with_failure(x));
    }

    pub fn clone_with_failure(self: *Cpu, x: anytype) core.Outcome(owning.Matrix(core.ScalarOf(@TypeOf(x)))) {
        const T = core.ScalarOf(@TypeOf(x));
        const xv = view.matViewFromAny(x);
        return core.intoOutcome(owning.Matrix(T), .clone, alloc_exec, ops.cloneAlloc(T, self.alloc, xv));
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

        const e = self.exec();
        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);
        return core.intoOutcome(owning.Matrix(T), .add, e, ops.addAllocEx(T, self.alloc, e, av, bv));
    }

    pub fn sub(self: *Cpu, a: anytype, b: anytype) core.LaError!owning.Matrix(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(owning.Matrix(T), self.sub_with_failure(a, b));
    }

    pub fn sub_with_failure(self: *Cpu, a: anytype, b: anytype) core.Outcome(owning.Matrix(core.ScalarOf(@TypeOf(a)))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("sub expects matching scalar types");

        const e = self.exec();
        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);
        return core.intoOutcome(owning.Matrix(T), .sub, e, ops.subAllocEx(T, self.alloc, e, av, bv));
    }

    pub fn mul(self: *Cpu, a: anytype, b: anytype) core.LaError!owning.Matrix(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(owning.Matrix(T), self.mul_with_failure(a, b));
    }

    pub fn mul_with_failure(self: *Cpu, a: anytype, b: anytype) core.Outcome(owning.Matrix(core.ScalarOf(@TypeOf(a)))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("mul expects matching scalar types");

        const e = self.exec();
        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);
        return core.intoOutcome(owning.Matrix(T), .mul, e, ops.mulAllocEx(T, self.alloc, e, av, bv));
    }

    pub fn div(self: *Cpu, a: anytype, b: anytype) core.LaError!owning.Matrix(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(owning.Matrix(T), self.div_with_failure(a, b));
    }

    pub fn div_with_failure(self: *Cpu, a: anytype, b: anytype) core.Outcome(owning.Matrix(core.ScalarOf(@TypeOf(a)))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("div expects matching scalar types");

        const e = self.exec();
        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);
        return core.intoOutcome(owning.Matrix(T), .div, e, ops.divAllocEx(T, self.alloc, e, av, bv));
    }

    pub fn relu(self: *Cpu, x: anytype) core.LaError!owning.Matrix(core.ScalarOf(@TypeOf(x))) {
        const T = core.ScalarOf(@TypeOf(x));
        return core.unwrapOutcome(owning.Matrix(T), self.relu_with_failure(x));
    }

    pub fn relu_with_failure(self: *Cpu, x: anytype) core.Outcome(owning.Matrix(core.ScalarOf(@TypeOf(x)))) {
        const T = core.ScalarOf(@TypeOf(x));
        const e = self.exec();
        const xv = view.matViewFromAny(x);
        return core.intoOutcome(owning.Matrix(T), .relu, e, ops.reluAllocEx(T, self.alloc, e, xv));
    }

    pub fn matmul(self: *Cpu, a: anytype, b: anytype) core.LaError!owning.Matrix(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(owning.Matrix(T), self.matmul_with_failure(a, b));
    }

    pub fn matmul_with_failure(self: *Cpu, a: anytype, b: anytype) core.Outcome(owning.Matrix(core.ScalarOf(@TypeOf(a)))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("matmul expects matching scalar types");

        const e = self.exec();
        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);
        return core.intoOutcome(owning.Matrix(T), .matmul, e, ops.matmulAllocEx(T, self.alloc, e, av, bv));
    }

    pub fn add_into(self: *Cpu, out: anytype, a: anytype, b: anytype) core.LaError!void {
        return core.unwrapOutcome(void, self.add_into_with_failure(out, a, b));
    }

    pub fn add_into_with_failure(self: *Cpu, out: anytype, a: anytype, b: anytype) core.Outcome(void) {
        const e = self.exec();
        const result: core.LaError!void = blk: {
            const OutT = core.ScalarOf(@TypeOf(out));
            const A = core.ScalarOf(@TypeOf(a));
            const B = core.ScalarOf(@TypeOf(b));
            if (OutT != A or A != B) @compileError("add_into expects matching scalar types");

            const outv = view.matMutViewFromAny(out);
            const av = view.matViewFromAny(a);
            const bv = view.matViewFromAny(b);
            break :blk ops.addIntoEx(OutT, e, outv, av, bv);
        };
        return core.intoOutcome(void, .add, e, result);
    }

    pub fn sub_into(self: *Cpu, out: anytype, a: anytype, b: anytype) core.LaError!void {
        return core.unwrapOutcome(void, self.sub_into_with_failure(out, a, b));
    }

    pub fn sub_into_with_failure(self: *Cpu, out: anytype, a: anytype, b: anytype) core.Outcome(void) {
        const e = self.exec();
        const result: core.LaError!void = blk: {
            const OutT = core.ScalarOf(@TypeOf(out));
            const A = core.ScalarOf(@TypeOf(a));
            const B = core.ScalarOf(@TypeOf(b));
            if (OutT != A or A != B) @compileError("sub_into expects matching scalar types");

            const outv = view.matMutViewFromAny(out);
            const av = view.matViewFromAny(a);
            const bv = view.matViewFromAny(b);
            break :blk ops.subIntoEx(OutT, e, outv, av, bv);
        };
        return core.intoOutcome(void, .sub, e, result);
    }

    pub fn mul_into(self: *Cpu, out: anytype, a: anytype, b: anytype) core.LaError!void {
        return core.unwrapOutcome(void, self.mul_into_with_failure(out, a, b));
    }

    pub fn mul_into_with_failure(self: *Cpu, out: anytype, a: anytype, b: anytype) core.Outcome(void) {
        const e = self.exec();
        const result: core.LaError!void = blk: {
            const OutT = core.ScalarOf(@TypeOf(out));
            const A = core.ScalarOf(@TypeOf(a));
            const B = core.ScalarOf(@TypeOf(b));
            if (OutT != A or A != B) @compileError("mul_into expects matching scalar types");

            const outv = view.matMutViewFromAny(out);
            const av = view.matViewFromAny(a);
            const bv = view.matViewFromAny(b);
            break :blk ops.mulIntoEx(OutT, e, outv, av, bv);
        };
        return core.intoOutcome(void, .mul, e, result);
    }

    pub fn div_into(self: *Cpu, out: anytype, a: anytype, b: anytype) core.LaError!void {
        return core.unwrapOutcome(void, self.div_into_with_failure(out, a, b));
    }

    pub fn div_into_with_failure(self: *Cpu, out: anytype, a: anytype, b: anytype) core.Outcome(void) {
        const e = self.exec();
        const result: core.LaError!void = blk: {
            const OutT = core.ScalarOf(@TypeOf(out));
            const A = core.ScalarOf(@TypeOf(a));
            const B = core.ScalarOf(@TypeOf(b));
            if (OutT != A or A != B) @compileError("div_into expects matching scalar types");

            const outv = view.matMutViewFromAny(out);
            const av = view.matViewFromAny(a);
            const bv = view.matViewFromAny(b);
            break :blk ops.divIntoEx(OutT, e, outv, av, bv);
        };
        return core.intoOutcome(void, .div, e, result);
    }

    pub fn matmul_into(self: *Cpu, out: anytype, a: anytype, b: anytype) core.LaError!void {
        return core.unwrapOutcome(void, self.matmul_into_with_failure(out, a, b));
    }

    pub fn matmul_into_with_failure(self: *Cpu, out: anytype, a: anytype, b: anytype) core.Outcome(void) {
        const e = self.exec();
        const result: core.LaError!void = blk: {
            const OutT = core.ScalarOf(@TypeOf(out));
            const A = core.ScalarOf(@TypeOf(a));
            const B = core.ScalarOf(@TypeOf(b));
            if (OutT != A or A != B) @compileError("matmul_into expects matching scalar types");

            const outv = view.matMutViewFromAny(out);
            const av = view.matViewFromAny(a);
            const bv = view.matViewFromAny(b);
            break :blk ops.matmulIntoEx(OutT, e, outv, av, bv);
        };
        return core.intoOutcome(void, .matmul, e, result);
    }

    pub fn gemm_into(self: *Cpu, comptime T: type, out: anytype, a: anytype, b: anytype, gemm_opts: core.GemmOpts(T)) core.LaError!void {
        return core.unwrapOutcome(void, self.gemm_into_with_failure(T, out, a, b, gemm_opts));
    }

    pub fn gemm_into_with_failure(self: *Cpu, comptime T: type, out: anytype, a: anytype, b: anytype, gemm_opts: core.GemmOpts(T)) core.Outcome(void) {
        const OutT = core.ScalarOf(@TypeOf(out));
        const A = core.ScalarOf(@TypeOf(a));
        const B = core.ScalarOf(@TypeOf(b));
        if (OutT != T or A != T or B != T) @compileError("gemm_into expects T to match out/a/b scalar types");

        const e = self.exec();
        const outv = view.matMutViewFromAny(out);
        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);
        return core.intoOutcome(void, .matmul, e, ops.gemmIntoEx(T, e, outv, av, bv, gemm_opts));
    }

    pub fn relu_inplace(self: *Cpu, x: anytype) core.LaError!void {
        return core.unwrapOutcome(void, self.relu_inplace_with_failure(x));
    }

    pub fn relu_inplace_with_failure(self: *Cpu, x: anytype) core.Outcome(void) {
        const e = self.exec();
        const result: core.LaError!void = blk: {
            const T = core.ScalarOf(@TypeOf(x));
            const xv = view.matMutViewFromAny(x);
            break :blk ops.reluInplaceEx(T, e, xv);
        };
        return core.intoOutcome(void, .relu, e, result);
    }

    pub fn sum(self: *Cpu, x: anytype) core.LaError!core.ScalarOf(@TypeOf(x)) {
        const T = core.ScalarOf(@TypeOf(x));
        return core.unwrapOutcome(T, self.sum_with_failure(x));
    }

    pub fn sum_with_failure(self: *Cpu, x: anytype) core.Outcome(core.ScalarOf(@TypeOf(x))) {
        const T = core.ScalarOf(@TypeOf(x));
        const e = self.exec();
        return core.intoOutcome(T, .sum, e, ops.sumEx(T, e, view.matViewFromAny(x)));
    }

    pub fn max(self: *Cpu, x: anytype) core.LaError!core.ScalarOf(@TypeOf(x)) {
        const T = core.ScalarOf(@TypeOf(x));
        return core.unwrapOutcome(T, self.max_with_failure(x));
    }

    pub fn max_with_failure(self: *Cpu, x: anytype) core.Outcome(core.ScalarOf(@TypeOf(x))) {
        const T = core.ScalarOf(@TypeOf(x));
        const e = self.exec();
        return core.intoOutcome(T, .max, e, ops.maxEx(T, e, view.matViewFromAny(x)));
    }

    pub fn norm2(self: *Cpu, x: anytype) core.LaError!core.ScalarOf(@TypeOf(x)) {
        const T = core.ScalarOf(@TypeOf(x));
        return core.unwrapOutcome(T, self.norm2_with_failure(x));
    }

    pub fn norm2_with_failure(self: *Cpu, x: anytype) core.Outcome(core.ScalarOf(@TypeOf(x))) {
        const T = core.ScalarOf(@TypeOf(x));
        const e = self.exec();
        return core.intoOutcome(T, .norm2, e, ops.norm2Ex(T, e, view.matViewFromAny(x)));
    }

    pub fn dot(self: *Cpu, a: anytype, b: anytype) core.LaError!core.ScalarOf(@TypeOf(a)) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(T, self.dot_with_failure(a, b));
    }

    pub fn dot_with_failure(self: *Cpu, a: anytype, b: anytype) core.Outcome(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("dot expects matching scalar types");

        const e = self.exec();
        return core.intoOutcome(T, .dot, e, ops.dotEx(T, e, view.vecViewFromAny(a), view.vecViewFromAny(b)));
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
        return core.intoOutcome(void, .frame_reset, core.Exec{}, result);
    }

    pub fn mat(self: *Frame, comptime T: type, rows: usize, cols: usize) core.LaError!*owning.TempMat(T) {
        return core.unwrapOutcome(*owning.TempMat(T), self.mat_with_failure(T, rows, cols));
    }

    pub fn mat_with_failure(self: *Frame, comptime T: type, rows: usize, cols: usize) core.Outcome(*owning.TempMat(T)) {
        return core.intoOutcome(*owning.TempMat(T), .frame_mat_alloc, core.Exec{}, self.allocTempMat(T, rows, cols));
    }

    pub fn vec(self: *Frame, comptime T: type, len: usize) core.LaError!*owning.TempVec(T) {
        return core.unwrapOutcome(*owning.TempVec(T), self.vec_with_failure(T, len));
    }

    pub fn vec_with_failure(self: *Frame, comptime T: type, len: usize) core.Outcome(*owning.TempVec(T)) {
        return core.intoOutcome(*owning.TempVec(T), .frame_vec_alloc, core.Exec{}, self.allocTempVec(T, len));
    }

    pub fn add(self: *Frame, a: anytype, b: anytype) core.LaError!*owning.TempMat(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(*owning.TempMat(T), self.add_with_failure(a, b));
    }

    pub fn add_with_failure(self: *Frame, a: anytype, b: anytype) core.Outcome(*owning.TempMat(core.ScalarOf(@TypeOf(a)))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("frame.add expects matching scalar types");

        const e = self.cpu.exec();
        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);

        const result: core.LaError!*owning.TempMat(T) = blk: {
            var out = self.allocTempMat(T, av.rows, av.cols) catch |err| break :blk err;
            ops.addIntoEx(T, e, out.asMatMutView(), av, bv) catch |err| break :blk err;
            break :blk out;
        };
        return core.intoOutcome(*owning.TempMat(T), .add, e, result);
    }

    pub fn sub(self: *Frame, a: anytype, b: anytype) core.LaError!*owning.TempMat(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(*owning.TempMat(T), self.sub_with_failure(a, b));
    }

    pub fn sub_with_failure(self: *Frame, a: anytype, b: anytype) core.Outcome(*owning.TempMat(core.ScalarOf(@TypeOf(a)))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("frame.sub expects matching scalar types");

        const e = self.cpu.exec();
        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);

        const result: core.LaError!*owning.TempMat(T) = blk: {
            var out = self.allocTempMat(T, av.rows, av.cols) catch |err| break :blk err;
            ops.subIntoEx(T, e, out.asMatMutView(), av, bv) catch |err| break :blk err;
            break :blk out;
        };
        return core.intoOutcome(*owning.TempMat(T), .sub, e, result);
    }

    pub fn mul(self: *Frame, a: anytype, b: anytype) core.LaError!*owning.TempMat(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(*owning.TempMat(T), self.mul_with_failure(a, b));
    }

    pub fn mul_with_failure(self: *Frame, a: anytype, b: anytype) core.Outcome(*owning.TempMat(core.ScalarOf(@TypeOf(a)))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("frame.mul expects matching scalar types");

        const e = self.cpu.exec();
        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);

        const result: core.LaError!*owning.TempMat(T) = blk: {
            var out = self.allocTempMat(T, av.rows, av.cols) catch |err| break :blk err;
            ops.mulIntoEx(T, e, out.asMatMutView(), av, bv) catch |err| break :blk err;
            break :blk out;
        };
        return core.intoOutcome(*owning.TempMat(T), .mul, e, result);
    }

    pub fn div(self: *Frame, a: anytype, b: anytype) core.LaError!*owning.TempMat(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(*owning.TempMat(T), self.div_with_failure(a, b));
    }

    pub fn div_with_failure(self: *Frame, a: anytype, b: anytype) core.Outcome(*owning.TempMat(core.ScalarOf(@TypeOf(a)))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("frame.div expects matching scalar types");

        const e = self.cpu.exec();
        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);

        const result: core.LaError!*owning.TempMat(T) = blk: {
            var out = self.allocTempMat(T, av.rows, av.cols) catch |err| break :blk err;
            ops.divIntoEx(T, e, out.asMatMutView(), av, bv) catch |err| break :blk err;
            break :blk out;
        };
        return core.intoOutcome(*owning.TempMat(T), .div, e, result);
    }

    pub fn relu(self: *Frame, x: anytype) core.LaError!*owning.TempMat(core.ScalarOf(@TypeOf(x))) {
        const T = core.ScalarOf(@TypeOf(x));
        return core.unwrapOutcome(*owning.TempMat(T), self.relu_with_failure(x));
    }

    pub fn relu_with_failure(self: *Frame, x: anytype) core.Outcome(*owning.TempMat(core.ScalarOf(@TypeOf(x)))) {
        const T = core.ScalarOf(@TypeOf(x));
        const e = self.cpu.exec();
        const xv = view.matViewFromAny(x);

        const result: core.LaError!*owning.TempMat(T) = blk: {
            var out = self.allocTempMat(T, xv.rows, xv.cols) catch |err| break :blk err;
            var outm = out.asMatMutView();

            var i: usize = 0;
            while (i < xv.rows) : (i += 1) {
                var j: usize = 0;
                while (j < xv.cols) : (j += 1) {
                    outm.set(i, j, xv.get(i, j));
                }
            }

            ops.reluInplaceEx(T, e, outm) catch |err| break :blk err;
            break :blk out;
        };
        return core.intoOutcome(*owning.TempMat(T), .relu, e, result);
    }

    pub fn matmul(self: *Frame, a: anytype, b: anytype) core.LaError!*owning.TempMat(core.ScalarOf(@TypeOf(a))) {
        const T = core.ScalarOf(@TypeOf(a));
        return core.unwrapOutcome(*owning.TempMat(T), self.matmul_with_failure(a, b));
    }

    pub fn matmul_with_failure(self: *Frame, a: anytype, b: anytype) core.Outcome(*owning.TempMat(core.ScalarOf(@TypeOf(a)))) {
        const T = core.ScalarOf(@TypeOf(a));
        const U = core.ScalarOf(@TypeOf(b));
        if (T != U) @compileError("frame.matmul expects matching scalar types");

        const e = self.cpu.exec();
        const av = view.matViewFromAny(a);
        const bv = view.matViewFromAny(b);

        const result: core.LaError!*owning.TempMat(T) = blk: {
            var out = self.allocTempMat(T, av.rows, bv.cols) catch |err| break :blk err;
            ops.matmulIntoEx(T, e, out.asMatMutView(), av, bv) catch |err| break :blk err;
            break :blk out;
        };
        return core.intoOutcome(*owning.TempMat(T), .matmul, e, result);
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
