const ns = @import("nullspace");

pub const CudaContext = struct {
    pub fn init() ns.LaError!CudaContext {
        return error.BackendUnavailable;
    }

    pub fn deinit(self: *CudaContext) void {
        _ = self;
    }

    pub fn uploadMat(self: *CudaContext, comptime T: type, src: ns.MatView(T)) ns.LaError!DeviceMatrix(T) {
        _ = self;
        _ = src;
        return error.BackendUnavailable;
    }

    pub fn downloadMatInto(self: *CudaContext, comptime T: type, dst: ns.MatMutView(T), src: DeviceMatView(T)) ns.LaError!void {
        _ = self;
        _ = dst;
        _ = src;
        return error.BackendUnavailable;
    }

    pub fn matmulInto(
        self: *CudaContext,
        comptime T: type,
        out: DeviceMatMutView(T),
        a: DeviceMatView(T),
        b: DeviceMatView(T),
    ) ns.LaError!void {
        _ = self;
        _ = out;
        _ = a;
        _ = b;
        return error.BackendUnavailable;
    }
};

pub fn DeviceMatView(comptime T: type) type {
    return struct {
        pub const Scalar = T;

        device_ptr: usize,
        rows: usize,
        cols: usize,
        row_stride: usize,
        col_stride: usize,
    };
}

pub fn DeviceMatMutView(comptime T: type) type {
    return struct {
        pub const Scalar = T;

        device_ptr: usize,
        rows: usize,
        cols: usize,
        row_stride: usize,
        col_stride: usize,

        pub fn asConst(self: @This()) DeviceMatView(T) {
            return .{
                .device_ptr = self.device_ptr,
                .rows = self.rows,
                .cols = self.cols,
                .row_stride = self.row_stride,
                .col_stride = self.col_stride,
            };
        }
    };
}

pub fn DeviceMatrix(comptime T: type) type {
    return struct {
        pub const Scalar = T;

        device_ptr: usize = 0,
        rows: usize,
        cols: usize,
        row_stride: usize,
        col_stride: usize,

        pub fn asMatView(self: @This()) DeviceMatView(T) {
            return .{
                .device_ptr = self.device_ptr,
                .rows = self.rows,
                .cols = self.cols,
                .row_stride = self.row_stride,
                .col_stride = self.col_stride,
            };
        }

        pub fn asMatMutView(self: *@This()) DeviceMatMutView(T) {
            return .{
                .device_ptr = self.device_ptr,
                .rows = self.rows,
                .cols = self.cols,
                .row_stride = self.row_stride,
                .col_stride = self.col_stride,
            };
        }

        pub fn deinit(self: *@This(), ctx: *CudaContext) void {
            _ = ctx;
            self.* = .{
                .device_ptr = 0,
                .rows = 0,
                .cols = 0,
                .row_stride = 0,
                .col_stride = 0,
            };
        }
    };
}
