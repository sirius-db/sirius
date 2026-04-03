# Disk Fallback for Sirius Downgrade Path

## What This Is

Integrating cuCascade's new disk I/O pipeline backend (from NVIDIA/cuCascade PR #96) into Sirius's downgrade system so that when GPU memory is under pressure and host memory is also full, data can be spilled to disk instead of failing. Data is read back directly to GPU via the pipeline backend when a pipeline task needs it.

## Core Value

When host memory is exhausted during GPU→HOST downgrade, queries must not fail — data spills to disk transparently and is read back on demand.

## Requirements

### Validated

- ✓ GPU→HOST downgrade path — existing (via `downgrade_task.cpp`)
- ✓ cuCascade disk tier infrastructure — existing (disk_data_representation, disk_access_limiter, converter registry)
- ✓ Pipeline I/O backend — existing in cuCascade submodule (double-buffered pinned host pipelining)
- ✓ Representation converters — existing (GPU↔DISK, HOST↔DISK registered via `register_builtin_converters`)

### Active

- [ ] GPU→DISK fallback when HOST reservation fails during downgrade
- [ ] Disk→GPU on-demand read-back when pipeline tasks need disk-resident data
- [ ] Disk tier configuration via Sirius `.cfg` config file (mount path, capacity)
- [ ] Pipeline backend as default I/O backend for disk operations
- [ ] Converter registry initialization with pipeline backend

### Out of Scope

- HOST→DISK proactive spilling — not needed; only fall back to disk when HOST reservation fails
- Proactive disk→GPU/HOST upgrade — data stays on disk until a pipeline task explicitly needs it
- KvikIO or GDS backends — pipeline backend is the default; other backends are not exposed via config
- Multi-disk support — single disk mount path for v1

## Context

- cuCascade PR #96 (`feature/file-downgrade`) is already pulled into the local `cucascade/` submodule
- The pipeline backend uses double-buffered 64 MB pinned host memory to overlap PCIe D2H transfers with NVMe I/O — no explicit HOST hop needed from Sirius's perspective
- Current downgrade path: `downgrade_task::execute()` requests `Tier::HOST` reservation → on failure throws `rmm::out_of_memory` → caught and logged as non-fatal → batch skipped
- Data batches use `convert_to<T>()` with the converter registry to change tier representations
- Pipeline tasks already handle HOST-resident data (upgrade HOST→GPU before execution); the same pattern extends to DISK-resident data (upgrade DISK→GPU)
- `disk_data_representation` is RAII — destructor deletes the backing file automatically
- Disk file format: 32-byte header + serialized column metadata + 4KB-aligned data, supporting all cuDF column types

## Constraints

- **cuCascade API**: Must use the existing converter registry and `disk_data_representation` — no custom disk I/O
- **Thread safety**: Downgrade tasks run on a bounded thread pool; disk file path generation must be thread-safe (cuCascade's `generate_disk_file_path()` handles this)
- **CUDA stream ordering**: Pipeline backend handles stream synchronization internally; callers must pass valid streams
- **Config system**: Sirius uses libconfig++ (`.cfg` files) for runtime configuration — new disk settings must follow existing patterns in `src/config.cpp`

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Pipeline backend as default | Double-buffered design gives best throughput for sequential large writes; internally handles HOST buffering | — Pending |
| GPU→DISK only (no HOST→DISK) | Minimal scope; HOST→DISK adds monitor complexity without clear benefit for current workloads | — Pending |
| On-demand read-back (no proactive upgrade) | Matches existing HOST→GPU pattern; avoids wasted I/O if data never needed again | — Pending |
| Config via .cfg file (not env vars) | Consistent with existing Sirius configuration pattern | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-02 after initialization*
