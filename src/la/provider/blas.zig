const core = @import("../core.zig");
const view = @import("../view.zig");

// Placeholder provider: compile/link boundary exists, but BLAS kernels are not wired yet.
pub const is_available = false;

pub fn canUse(
    comptime T: type,
    out: view.MatMutView(T),
    a: view.MatView(T),
    b: view.MatView(T),
) bool {
    return is_available and
        core.isFloatType(T) and
        out.isContiguousRowMajor() and
        a.isContiguousRowMajor() and
        b.isContiguousRowMajor();
}

pub fn matmulInto(
    comptime T: type,
    out: view.MatMutView(T),
    a: view.MatView(T),
    b: view.MatView(T),
) core.LaError!void {
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
