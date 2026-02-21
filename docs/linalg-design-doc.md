# Nullspace Linear Algebra Foundations

**Status:** Active design contract
**Scope:** Conceptual foundations for the public API and core internals
**Updated:** 2026-02-21

## 1. Purpose

`nullspace` is a Zig-first linear algebra library designed around explicit memory and explicit execution.

This document is intentionally concise: it defines the stable mental model and design constraints, not an exhaustive API reference.

## 2. Core Foundations

1. Views are the compute substrate.
2. Owning containers exist for persistence and ownership transfer.
3. Allocation is explicit in API shape.
4. CPU is the default and only first-class execution domain today.
5. Safety checks are deterministic and fail-closed.
6. Advanced behavior is opt-in and separated from common paths.

## 3. Architectural Model

- `view.zig`: non-owning descriptors (`MatView`, `MatMutView`, `VecView`, `VecMutView`).
- `owning.zig`: owning memory (`Matrix(T)`, `Vector(T)`) and frame-bound temp wrappers (`TempMat`, `TempVec`).
- `contract.zig`: validation and alias/shape/stride rules.
- `backend/cpu_scalar.zig`: scalar reference kernels (authoritative behavior).
- `backend/cpu_simd.zig`: SIMD entry points (currently capability-gated/stubbed).
- `ops.zig`: internal dispatch layer selected by `Exec`.
- `cpu.zig`: user-facing CPU and frame façade.

The control flow is: façade -> views -> contract checks -> dispatch -> kernel.

## 4. API Posture

Public API favors minimal common calls and explicit advanced calls:

- Common path: `matmul_into(out, a, b)` computes overwrite semantics (`out = a*b`).
- Advanced path: `gemm_into(T, out, a, b, GemmOpts(T))` enables `alpha/beta` accumulation.
- Transpose is a view transform (`.t()`), not a matmul option flag.
- Public user-facing names stay `snake_case`.

Removed by design from common calls:

- Backend enums in user call signatures.
- Per-op option structs for elementwise/reduction/matmul defaults.

## 5. Allocation and Lifetime Contracts

- `*_into` and `*_inplace` do not allocate by contract.
- Allocating APIs are explicit (`mat`, `vec`, `add`, `matmul`, etc.).
- `Cpu` allocations are persistent and caller-owned (`deinit` with the same allocator).
- `Frame` allocations are arena-scoped; lifetime ends at `frame.reset()` or `frame.deinit()`.
- `TempMat`/`TempVec` are frame-constructed opaque wrappers to make temporary lifetime explicit.

## 6. Type and Safety Boundaries

- Numeric scalar constraints are compile-time enforced.
- Mutability is enforced at type boundaries (`MatView` vs `MatMutView`, `VecView` vs `VecMutView`).
- `Failure` reports the requested execution intent (`Exec`), not a hidden resolved backend.
- Alias rules are operation-specific:
  - Elementwise `*_into`: exact alias is allowed; partial overlap is rejected.
  - `matmul_into`/`gemm_into`: overlap with inputs is rejected.
- Stride/overlap arithmetic overflows fail closed as `error.InvalidStride`.

## 7. Execution Model

`Exec` is the execution request contract:

- `simd: .auto | .scalar | .simd`
- `threads: .single | .multi` (policy field retained for future implementation)
- `device: .cpu` (reserved extension point)

Dispatch policy:

- `.auto`: choose best available CPU implementation.
- `.scalar`: require scalar path.
- `.simd`: require SIMD support or return `error.BackendUnavailable`.

## 8. Extension Strategy (Deferred by Design)

- BLAS and CUDA are not part of the default public execution vocabulary.
- CUDA is treated as a separate memory domain with explicit transfer APIs (future module direction), not as a runtime backend enum on host views.
- Any future accelerator integration must preserve the no-hidden-allocation contract for `_into`/`_inplace` APIs.

## 9. Design Guardrails for Future Changes

When evolving the API, preserve these invariants:

1. Keep common operations option-free and concept-light.
2. Push rare semantics into explicitly named advanced APIs.
3. Preserve explicit ownership and allocator provenance.
4. Prefer compile-time type errors for type/shape-policy misuse where practical.
5. Keep failure metadata about caller intent stable (`OpKind`, `Exec`, `LaError`).
6. Avoid introducing abstraction layers that hide memory movement or allocation.

This is the project’s guiding foundation; implementation details can evolve, but these constraints should remain stable.
