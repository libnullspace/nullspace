# AGENTS

This file defines contributor/agent rules for work in this repository.

## Scope

- Target project: `nullspace` Zig linear algebra library
- Public code lives in `src/`, `build.zig`, `build.zig.zon`, and top-level docs

## Hard Rules

- Preserve explicit allocation behavior; do not introduce hidden allocations in `_into` / `_inplace` APIs.

## Engineering Expectations

- Prefer root-cause fixes over local patches.
- Keep API contracts explicit and consistent across allocating and non-allocating variants.
- When adding mutable APIs, enforce mutability at the type boundary (`*T` vs `*const T`) rather than relying on caller discipline.
- Add tests for behavior changes and edge cases (shape mismatch, aliasing, backend selection, allocator/lifetime paths).
- When public APIs or behavior change, update the relevant top-level docs (`README.md` and public docs under `docs/`).
- Prefer explicit error propagation; avoid `catch unreachable` on allocator/lifetime paths.

## Formatting

- Use `zig fmt` for all changed `.zig` and `.zon` files before finishing.
- For verification, prefer `zig fmt --check src build.zig build.zig.zon` and run `zig fmt` if any files are reported.

## Workflow

- Make targeted changes.
- Run `zig build test` and `zig build`.

### Testing Minimums

- Docs-only changes: no build required unless code snippets or version requirements changed.
- Code changes without public API changes: run `zig build test` and `zig build`.
- Public API/behavior changes: run `zig build test` and `zig build`, and add/update targeted tests.

## Review Standard

- Prioritize correctness and safety over convenience.
- Include file/line references for findings or fixes.
- Call out assumptions and unresolved risks explicitly.
