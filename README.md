# nullspace

`nullspace` is a Zig linear algebra library focused on explicit memory control, predictable behavior, and a clean CPU-first API.

## Status

The project is functional and test-covered for core matrix/vector operations, with CPU scalar kernels implemented today and explicit execution/contract checks wired through the CPU API.

## Highlights

- View-first compute model: `MatView`, `MatMutView`, `VecView`, `VecMutView`
- View transpose helpers: `MatView.t()` and `MatMutView.t()`
- Owning containers: `Matrix(T)`, `Vector(T)`
- Ergonomic CPU facade: `cpu(alloc)` with allocating, `_into`, and `_inplace` APIs
- Advanced accumulation API: `gemm_into(T, out, a, b, GemmOpts(T))`
- Frame-scoped temporary allocations via arena (`Frame`)
- Structured error context via `Outcome(T)` / `Failure` (including requested `Exec`)

## Requirements

- Zig `>= 0.15.1` (see `build.zig.zon`)

## Build And Run

```sh
zig build
zig build run
zig build test
```

## Quick Example

```zig
const std = @import("std");
const ns = @import("nullspace");

pub fn main() !void {
    var cpu = ns.cpu(std.heap.page_allocator);
    defer cpu.deinit();

    var a = try cpu.mat(f32, 2, 2);
    defer a.deinit(cpu.alloc);
    var b = try cpu.mat(f32, 2, 2);
    defer b.deinit(cpu.alloc);

    var av = a.asMatMutView();
    av.set(0, 0, 1); av.set(0, 1, 2);
    av.set(1, 0, 3); av.set(1, 1, 4);

    var bv = b.asMatMutView();
    bv.set(0, 0, 5); bv.set(0, 1, 6);
    bv.set(1, 0, 7); bv.set(1, 1, 8);

    var out = try cpu.matmul(a, b);
    defer out.deinit(cpu.alloc);

    std.debug.print("C[0,0]={d}\n", .{out.asMatView().get(0, 0)});
}
```

## API Shape

- Allocating: `add`, `sub`, `mul`, `div`, `relu`, `matmul`, `clone`, `mat`, `vec`
- Non-allocating output: `add_into`, `sub_into`, `mul_into`, `div_into`, `matmul_into`, `gemm_into`
- In-place: `relu_inplace`
- Reductions: `sum`, `max`, `dot`, `norm2`
- Execution knobs: `Cpu.simd`, `Cpu.threads`, `Exec`
- Temporary scope: `frame()`, `frame.reset()`, `frame.deinit()`

`matmul_into` is overwrite-only (`out = a*b`).
`gemm_into` is the accumulation form (`out = alpha*(a*b) + beta*out`).

## Backends

- Implemented: `cpu_scalar`
- Capability-gated (currently return `error.BackendUnavailable`): `cpu_simd`
- BLAS/CUDA are not part of the current public CPU-first surface

## Repo Notes

- Public source of truth is under `src/`, `build.zig`, and `build.zig.zon`.
