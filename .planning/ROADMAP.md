# Roadmap: Disk Fallback for Sirius Downgrade Path

## Overview

Two phases to integrate cuCascade's disk I/O pipeline backend into Sirius's downgrade system. Phase 1 wires disk tier configuration into the Sirius engine and converter registry. Phase 2 implements the runtime behavior: GPU->DISK fallback when HOST reservation fails, and DISK->GPU read-back when pipeline tasks need disk-resident data.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Disk Tier Wiring** - Configure disk memory space, capacity, and pipeline backend in Sirius engine (completed 2026-04-03)
- [x] **Phase 2: End-to-End Spill Flow** - GPU->DISK fallback during downgrade and DISK->GPU read-back during pipeline execution (completed 2026-04-03)

## Phase Details

### Phase 1: Disk Tier Wiring
**Goal**: Sirius engine creates and manages a disk memory space when disk config is present
**Depends on**: Nothing (first phase)
**Requirements**: CFG-01, CFG-02, CFG-03
**Success Criteria** (what must be TRUE):
  1. Sirius `.cfg` file accepts `disk_mount_path` and `disk_capacity` settings without error
  2. When disk config is present, engine startup creates a disk memory space and registers it with the reservation manager
  3. Converter registry is initialized with the pipeline backend at engine startup; GPU<->DISK and HOST<->DISK converters are available
**Plans:** 1/1 plans complete
Plans:
- [x] 01-01-PLAN.md — Rename disk config keys to user-friendly names and verify disk space creation + converter registry

### Phase 2: End-to-End Spill Flow
**Goal**: Queries survive GPU+HOST memory exhaustion by transparently spilling to disk and reading back on demand
**Depends on**: Phase 1
**Requirements**: DG-01, DG-02, DG-03, RB-01, RB-02
**Success Criteria** (what must be TRUE):
  1. When HOST reservation fails during GPU downgrade, data is written to disk instead of the batch being skipped
  2. A query that would previously fail with out-of-memory (GPU full + HOST full) now completes with correct results when disk is configured
  3. Disk fallback events are logged at INFO level with batch ID and data size
  4. Pipeline tasks consuming disk-resident batches convert DISK->GPU via the pipeline backend before execution (same observable pattern as HOST->GPU upgrade)
**Plans:** 2/2 plans complete
Plans:
- [x] 02-01-PLAN.md — Disk fallback in downgrade_task::execute() (DG-01, DG-02, DG-03)
- [x] 02-02-PLAN.md — DISK->GPU read-back test for pipeline tasks (RB-01, RB-02)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Disk Tier Wiring | 1/1 | Complete   | 2026-04-03 |
| 2. End-to-End Spill Flow | 2/2 | Complete   | 2026-04-03 |
