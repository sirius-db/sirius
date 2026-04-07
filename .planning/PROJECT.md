# Sirius Debug Utilities

## What This Is

A C++ debugging utility library for the Sirius GPU SQL engine that provides structured, human-readable data inspection functions for use during query debugging. These utilities replace ad-hoc log statements with formal, reusable functions that the `/validate` and `/runtime-errors` Claude Code skills can programmatically insert into operator code during diagnosis.

## Core Value

Enable fast, accurate identification of faulty operators by providing consistent, pretty-printed data inspection at any point in the GPU execution pipeline.

## Requirements

### Validated

- ✓ `debug_schema(batch)` prints column names, data types, null counts, and total row count via SIRIUS_LOG — Phase 1
- ✓ `debug_nulls(batch)` logs per-column null count and null percentage — Phase 1
- ✓ Infrastructure: stream-scoped sync, null-aware copy, [SIRIUS_DIAG] log routing, output buffering, try/catch wrapping — Phase 1

### Active

- [ ] `debug_head(batch, N)` prints first N rows of a data batch in aligned-column format (pandas-style) and CSV format via SIRIUS_LOG
- [ ] `debug_stats(batch)` prints per-column min, max, sum statistics via SIRIUS_LOG
- [ ] `debug_checksum(batch)` computes and logs per-column hash/fingerprint for cross-pipeline comparison
- [ ] `debug_diff(batch_a, batch_b)` compares two data batches and logs which rows and columns differ
- [ ] All utilities support the full set of Sirius-supported data types (INTEGER, BIGINT, FLOAT, DOUBLE, VARCHAR, DATE, TIMESTAMP, DECIMAL)
- [ ] Output is pretty-formatted and human-readable in log files
- [ ] `/validate` skill references these utilities for data validation instrumentation
- [ ] `/runtime-errors` skill references these utilities for crash/hang diagnosis

### Out of Scope

- Python wrappers for interactive debugging — C++ only for now
- Runtime performance profiling (use `/profile-analyzer` instead)
- Persistent data dumping to files (output goes through SIRIUS_LOG only)
- GUI or web-based data viewers

## Context

- Sirius already has basic `print_table_contents()` and `print_data_batch_contents()` in `src/include/print.hpp` / `src/cuda/print.cu` — these only support numeric/bool types and simple row dumps
- The `/validate` skill currently instructs Claude to manually construct ad-hoc `SIRIUS_LOG_TRACE("[SIRIUS_DIAG] operator_name checksum: sum={}, max={}, first_row={}", ...)` lines
- The `/runtime-errors` skill inserts targeted `SIRIUS_LOG_TRACE("[SIRIUS_DIAG] ...")` statements at suspected fault points
- Data batches wrap `cucascade::data_batch` → `gpu_table_representation` → `cudf::table`
- Data must be copied from GPU to host before printing
- Existing logging uses spdlog via `SIRIUS_LOG_*` macros defined in `src/include/log/logging.hpp`

## Constraints

- **Data types**: Must handle all Sirius-supported types: INT8/16/32/64, UINT8/16/32/64, FLOAT32/64, BOOL8, STRING, TIMESTAMP, DATE, DECIMAL
- **GPU→Host**: All print functions must copy data from GPU memory to host before formatting — keep copies minimal
- **Thread safety**: Functions may be called from GPU pipeline task threads — must be safe for concurrent use
- **Log integration**: All output via `SIRIUS_LOG_DEBUG` or `SIRIUS_LOG_TRACE` macros (controlled by `SIRIUS_LOG_LEVEL`)
- **Build system**: Must integrate with existing CMake CUDA build (separable compilation, C++20/CUDA 20)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Extend existing print.hpp/print.cu rather than new module | Builds on existing infrastructure, avoids file proliferation | -- Pending |
| Output via SIRIUS_LOG macros, not stderr | Keeps all debug output in the same log pipeline the skills already parse | -- Pending |
| Two output formats: aligned columns + CSV | Aligned for human readability in logs, CSV for programmatic comparison | -- Pending |
| Tag all debug output with `[SIRIUS_DIAG]` prefix | Skills can grep for diagnostic output vs normal logs | -- Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-07 after Phase 1 completion*
