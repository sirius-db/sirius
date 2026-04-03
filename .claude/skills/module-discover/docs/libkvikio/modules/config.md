# Configuration & Defaults

**Status**: UNUSED
**Path**: `kvikio/defaults.hpp`, `kvikio/compat_mode.hpp`, `kvikio/cufile/driver.hpp`, `kvikio/cufile/config.hpp`

## Summary
Global configuration singleton for kvikio. Controls compatibility mode (cuFile vs POSIX fallback), thread pool sizing, task sizes, GDS thresholds, bounce buffer sizes, HTTP settings, and Direct I/O behavior. Also provides cuFile driver property access.

## Key APIs
- `defaults` — Singleton with static getters/setters for all global settings
- `CompatMode` — Enum: `OFF` (enforce cuFile), `ON` (enforce POSIX), `AUTO` (try cuFile, fall back)
- `DriverProperties` — Query cuFile driver capabilities (GDS availability, version, cache sizes)
- `DriverInitializer` — RAII wrapper to open/close cuFile driver
- `getenv_or()` — Utility to read environment variables with defaults

### Key environment variables (read by `defaults`):
- `KVIKIO_COMPAT_MODE` — Set compatibility mode
- `KVIKIO_NTHREADS` — Thread pool size
- `KVIKIO_TASK_SIZE` — Default task size for parallel I/O
- `KVIKIO_GDS_THRESHOLD` — Minimum size to use GDS (below uses POSIX)
- `KVIKIO_BOUNCE_BUFFER_SIZE` — Bounce buffer size

## Potential Relevance
Low unless directly integrating kvikio for file I/O. The `CompatMode::AUTO` pattern is useful to know — it's how cudf's Parquet reader transparently uses GDS when available.
