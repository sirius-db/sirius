# Change Classification Guide

This guide maps source file paths to documentation files for the `/update-docs` skill.

## Source Path → Documentation File Mapping

| Source Path Pattern | Documentation File | What to Update |
|---|---|---|
| `src/op/sirius_physical_*.cpp` or `.hpp` | `operators.md` | Operator description, cuDF API usage, modes |
| `src/op/scan/` | `scan.md` | Scan tasks, executor, caching |
| `src/planner/` | `physical-plan-generation.md` | Logical-to-physical mappings, plan builders |
| `src/sirius_engine.cpp` (`initialize_internal`) | `physical-plan-generation.md` | Pipeline splitting rules |
| `src/pipeline/` | `pipeline-execution.md` | Executor, task scheduling, completion |
| `src/creator/` | `task-creator.md` | Task creation, hint chain |
| `src/op/sirius_physical_operator.cpp` (`get_next_task_hint`) | `task-creator.md` | Per-operator overrides table |
| `src/downgrade/` or `src/memory/` | `memory-management.md` | Memory tiers, reservations, downgrade |
| `src/expression_executor/` | `expression-executor.md` | Expression types, translator |
| `src/include/sirius_config.hpp` or `config.hpp` | `configuration.md` | Config params, SET vars |
| `src/include/data/` | `data-management.md` | Data batches, repos, ports |
| `src/sirius_interface.cpp` or `src/sirius_extension.cpp` | `execution-flow.md` | Entry points, lifecycle |
| `src/include/sirius_context.hpp` | `architecture-overview.md` | Ownership, thread model |

## PR Title Heuristics

| PR Title Pattern | Documentation File | Action |
|---|---|---|
| Contains "optim", "perf", "speed", "improve" | `optimizations.md` | Add new optimization entry |
| Contains "operator", "join", "aggregate", "sort", "filter" | `operators.md` | Update operator description |
| Contains "scan", "parquet", "cache" | `scan.md` | Update scan/caching docs |
| Contains "pipeline", "executor", "task" | `pipeline-execution.md` | Update execution docs |
| Contains "memory", "downgrade", "spill", "OOM" | `memory-management.md` | Update memory docs |
| Contains "config", "setting", "parameter" | `configuration.md` | Update config docs |

## Files to Skip

These file patterns generally don't require documentation updates:
- `test/` — test files
- `CMakeLists.txt`, `*.cmake` — build configuration
- `.github/` — CI/CD
- `third_party/` — external dependencies
- `cucascade/` — cuCascade library (unless API changes affect Sirius integration)
- `duckdb/` — DuckDB submodule
- `.pre-commit-config.yaml`, `.clang-format` — code style
- `scripts/` — utility scripts
- `docs/` — documentation itself (avoid circular updates)
