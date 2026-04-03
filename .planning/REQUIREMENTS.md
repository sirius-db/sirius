# Requirements: Disk Fallback for Sirius Downgrade Path

**Defined:** 2026-04-02
**Core Value:** When host memory is exhausted during GPU→HOST downgrade, queries must not fail — data spills to disk transparently and is read back on demand.

## v1 Requirements

### Downgrade Fallback

- [x] **DG-01**: When HOST reservation fails during GPU downgrade, system requests DISK tier reservation as fallback
- [x] **DG-02**: Downgrade task dynamically selects `host_data_representation` or `disk_data_representation` based on which tier the reservation was granted from
- [x] **DG-03**: Disk fallback is logged at INFO level with batch ID and data size

### Read-back

- [x] **RB-01**: Pipeline tasks can consume disk-resident data batches by converting DISK→GPU before execution (same pattern as existing HOST→GPU upgrade)
- [x] **RB-02**: DISK→GPU conversion uses the pipeline backend (double-buffered pinned host pipelining)

### Configuration

- [x] **CFG-01**: Sirius `.cfg` config file supports `disk_mount_path` (string) and `disk_capacity` (size) settings
- [x] **CFG-02**: When disk config is present, `sirius_context` creates a disk memory space and adds it to the reservation manager
- [x] **CFG-03**: Converter registry is initialized with pipeline backend at engine startup

## v2 Requirements

### Proactive Spilling

- **PS-01**: Monitor HOST tier pressure and proactively spill HOST data to disk before HOST is full
- **PS-02**: Proactive DISK→GPU/HOST upgrade when higher-tier memory frees up

### Advanced Config

- **AC-01**: Multiple disk mount paths for striping across NVMe drives
- **AC-02**: Environment variable overrides for disk settings (SIRIUS_DISK_PATH, SIRIUS_DISK_CAPACITY)
- **AC-03**: Configurable I/O backend selection (kvikIO, GDS, pipeline) via config file

## Out of Scope

| Feature | Reason |
|---------|--------|
| HOST→DISK proactive spilling | Adds monitor complexity; GPU→DISK fallback sufficient for v1 |
| Proactive disk→GPU upgrade | Data stays on disk until needed; avoids wasted I/O |
| KvikIO/GDS backend exposure | Pipeline backend is default; others can be added later |
| Multi-disk support | Single mount path sufficient for v1 |
| Fallback logging metrics/counters | Simple INFO logging sufficient for v1 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CFG-01 | Phase 1 | Complete |
| CFG-02 | Phase 1 | Complete |
| CFG-03 | Phase 1 | Complete |
| DG-01 | Phase 2 | Complete |
| DG-02 | Phase 2 | Complete |
| DG-03 | Phase 2 | Complete |
| RB-01 | Phase 2 | Complete |
| RB-02 | Phase 2 | Complete |

**Coverage:**
- v1 requirements: 8 total
- Mapped to phases: 8
- Unmapped: 0

---
*Requirements defined: 2026-04-02*
*Last updated: 2026-04-02 after roadmap creation*
