const std = @import("std");
const nullspace = @import("nullspace");

pub fn main() !void {
    var stdout_buffer: [256]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    var c = nullspace.cpu(std.heap.page_allocator);
    defer c.deinit();

    var m = try c.mat(f32, 1, 3);
    defer m.deinit(c.alloc);

    var mv = m.asMatMutView();
    mv.set(0, 0, 1);
    mv.set(0, 1, 2);
    mv.set(0, 2, 3);

    const total = try c.sum(m);
    try stdout.print("nullspace linalg ready, sum={d}\n", .{total});
    try stdout.flush();
}
