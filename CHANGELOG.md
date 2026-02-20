# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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
