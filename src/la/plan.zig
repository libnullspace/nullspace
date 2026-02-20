const core = @import("core.zig");

pub fn GemmPlan(comptime T: type) type {
    core.assertNumericType(T);
    return struct {
        pub const Scalar = T;

        m: usize,
        n: usize,
        k: usize,

        const Self = @This();

        pub fn init(m: usize, n: usize, k: usize) Self {
            return .{ .m = m, .n = n, .k = k };
        }
    };
}
