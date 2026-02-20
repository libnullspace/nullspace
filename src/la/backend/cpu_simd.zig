const core = @import("../core.zig");
const view = @import("../view.zig");

pub const is_available = false;

pub fn matmulInto(
    comptime T: type,
    out: view.MatMutView(T),
    a: view.MatView(T),
    b: view.MatView(T),
    opts: core.MatmulOpts,
) core.LaError!void {
    _ = out;
    _ = a;
    _ = b;
    _ = opts;
    return error.BackendUnavailable;
}
