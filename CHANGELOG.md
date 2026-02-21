# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

Refines the API around explicit execution intent and matrix multiplication semantics.

### Added

- `gemm_into` CPU API (`out = alpha * (a*b) + beta * out`) for explicit accumulation semantics.
- View transpose helpers: `MatView.t()` and `MatMutView.t()`.

### Changed

- Execution configuration now uses `Exec` (`simd`, `threads`, `device`) and typed `GemmOpts(T)`.
- `Failure` now reports requested execution intent through `fail.exec` instead of backend enum metadata.
- `matmul_into` is now overwrite-only (`out = a*b`), while accumulation lives in `gemm_into`.
- SIMD dispatch is now consistent across compute ops; requesting `.simd` returns `error.BackendUnavailable` when unavailable.
- Build flags now enforce real boundaries: BLAS is internal provider wiring, and CUDA is an optional `nullspace_cuda` module.

## [0.0.1] - 2026-02-20

Initial public release.

### Added

- Core linear algebra API with explicit allocation contracts.
- View types: `MatView`, `MatMutView`, `VecView`, `VecMutView`.
- Owning containers: `Matrix(T)`, `Vector(T)`.
- CPU facade with allocating, `_into`, and `_inplace` operations.
- Frame-scoped temporary allocation API via `Frame`.
- Structured error outcomes via `Outcome(T)` and `Failure`.
- Scalar CPU backend implementation (`cpu_scalar`) for core ops.
