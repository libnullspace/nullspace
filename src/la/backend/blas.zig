const core = @import("../core.zig");
const view = @import("../view.zig");
const cfg = @import("build_options");

pub const is_enabled = cfg.enable_blas;

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
