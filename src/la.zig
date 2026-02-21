const std = @import("std");

pub const core = @import("la/core.zig");
pub const view = @import("la/view.zig");
pub const contract = @import("la/contract.zig");
const owning = @import("la/owning.zig");
pub const ops = @import("la/ops.zig");
pub const cpu_api = @import("la/cpu.zig");
pub const plan = @import("la/plan.zig");

pub const backend = struct {
    pub const cpu_scalar = @import("la/backend/cpu_scalar.zig");
    pub const cpu_simd = @import("la/backend/cpu_simd.zig");
};

pub const LaError = core.LaError;
pub const SimdMode = core.SimdMode;
pub const Threading = core.Threading;
pub const Exec = core.Exec;
pub const OpKind = core.OpKind;
pub const Failure = core.Failure;
pub const Outcome = core.Outcome;
pub const ScalarOf = core.ScalarOf;
pub const GemmOpts = core.GemmOpts;

pub const VecView = view.VecView;
pub const VecMutView = view.VecMutView;
pub const MatView = view.MatView;
pub const MatMutView = view.MatMutView;

pub const Matrix = owning.Matrix;
pub const Vector = owning.Vector;
pub const TempMat = owning.TempMat;
pub const TempVec = owning.TempVec;

pub const Cpu = cpu_api.Cpu;
pub const Frame = cpu_api.Frame;
pub const GemmPlan = plan.GemmPlan;

pub fn cpu(alloc: std.mem.Allocator) Cpu {
    return cpu_api.cpu(alloc);
}
